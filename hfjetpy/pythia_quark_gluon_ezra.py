#!/usr/bin/env python
'''
Script for looking at the quark vs gluon dependence of substructure observables
Author: Ezra Lesser (elesser@berkeley.edu), some come from Beatrice L-G
'''

# Python3 standard library
import yaml
import argparse
import os
import sys
import array
import numpy as np
from array import array
import math
import cppyy
from enum import Enum

# Define cpp software to use
headers = [
    # FastJet
    "fastjet/PseudoJet.hh",
    "fastjet/JetDefinition.hh",
    "fastjet/ClusterSequence.hh",
    "fastjet/Selector.hh",
    "fastjet/JadePlugin.hh",

    # fj.contrib
    "fastjet/contrib/LundGenerator.hh",
    "fastjet/contrib/Recluster.hh",
    "fastjet/contrib/FlavInfo.hh",
    "fastjet/contrib/IFNPlugin.hh",
    "fastjet/contrib/GHSAlgo.hh",
    "fastjet/contrib/CMPPlugin.hh",

    # Pythia8
    "Pythia8/Pythia.h",
    "Pythia8/Event.h",
    "Pythia8/Info.h",
    #"Pythia8/PhaseSpace.h",
    #"Pythia8/TimeShower.h",
    #"Pythia8/Settings.h",

    # heppyy
    "fjext/fjtools.hh",
    #"groom/GroomerShop.hh",
    #"pythiaext/pythiahepmc.hh",
    "pythiafjext/pyfjtools.hh"
    ]
packs = [
    'fastjet',
    'pythia8',
    'heppyy'
    ]
libs  = [
    'fastjet',
    'fastjetplugins',
    #'LundPlane',
    #'RecursiveTools',
    'IFNPlugin', 'GHSAlgo', 'CMPPlugin', # jetflav
    'fastjetcontribfragile',  # all of fj.contrib
    'pythia8lhapdf6',
    'heppyy_fjext',
    #'heppyy_groom',
    #'heppyy_pythiaext',
    'heppyy_pythiafjext'
    ]

# Load packages using cppyy
from yasp.cppyyhelper import YaspCppyyHelper
YaspCppyyHelper().load(packs, libs, headers)
from cppyy.gbl import fastjet as fj
from cppyy.gbl import Pythia8 as pythia8
from cppyy.gbl import FJTools as fjext
#from cppyy.gbl import HepMCTools as pythiaext
from cppyy.gbl import pythiafjtools as pythiafjext
from cppyy.gbl.std import vector

# Use PyROOT (also operates on top of cppyy)
# https://root.cern/manual/python/
import ROOT
# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()
ROOT.TH3.SetDefaultSumw2()

from heppyy.pythia_util import configuration as pyconf

from hfjetpy.process import process_base


################################################################
class EMesonDecayChannel(Enum):
    kAnyDecay            = 0
    kUnknownDecay        = 1 #BIT(0)
    kDecayD0toKpi        = 2 #BIT(1)
    kDecayDStartoKpipi   = 3 #BIT(2)

class Promptness(Enum):
    kUnknown = 0
    kPrompt = 1
    kNonPrompt = 2

################################################################
class PythiaQuarkGluon(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaQuarkGluon, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.jetR_list = config["jetR"]

        self.user_seed = args.user_seed
        self.nev = args.nev

        self.noMPI = (bool)(1-args.MPIon)
        self.noISR = (bool)(1-args.ISRon)

        # self implemented variables to study
        self.charmdecaysOFF = (bool)(args.nocharmdecay) #charmdecaysOFF=True(F) when charmdecayon=1(0)
        print("charm decay input value", args.nocharmdecay)
        self.weighted = (bool)(args.weightON) #weightON=True(F) means turn weights on(off)
        self.leading_parton_pt_cut = args.leadingptcut
        self.replaceKPpairs = (bool)(args.replaceKP) #replaceKP=True(F) means turn k/pi pairs are('nt) replaced
        self.gg2ccbar = (bool)(args.onlygg2ccbar) #gg2ccbar=True means only run gg->ccbar process
        self.hardccbar = (bool)(args.onlyccbar) #hard2ccbar=True means only run hard->ccbar process
        self.Dstar = (bool)(args.DstarON) #Dstar=True means look at D* EEC, should be run with self.replaceKPpairs=True
        self.initscat = args.chinitscat #1=hard->ccbar, 2=gg->ccbar, 3=D0->Kpi channel, 4=hard->bbar w/ D0->Kpi
        self.D0wDstar = (bool)(args.D0withDstarON) #D0wDstar=True means looking at D-tagged jets including D0 from D*
        self.difNorm = (bool)(args.difNorm) #difNorm=True means normalize D* distribution with (D0+D*) jets
        self.softpion_action = args.softpion #1 = remove soft pion from D*, 2 = only pair soft pion with charged particles, 3 = only pair soft pion with D0, 4 = pair soft pion w everything
        self.use_ptRL = (bool)(args.giveptRL) #1=True=replace RL in THnSparse with pT*RL
        self.phimeson = (bool)(args.runphi) #1=don't let phi meson decay and look at its EEC

        # PDG ID values for quarks and gluons
        self.quark_pdg_ids = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]
        self.down_pdg_ids = [1, -1]
        self.up_pdg_ids = [2, -2]
        self.strange_pdg_ids = [3, -3]
        self.charm_pdg_ids = [4, -4]
        self.gluon_pdg_ids = [9, 21]
        self.beauty_pdg_ids = [5, -5]

        # hadron level - LHCb tracking restriction
        self.max_eta_hadron = 5
        self.min_eta_hadron = 2

        self.min_leading_track_pT = config["min_leading_track_pT"] if "min_leading_track_pT" in config else None

        self.pt_bins = array('d', list(range(5, 100, 5)) + list(range(100, 210, 10)))

        self.obs_bins_ang = np.concatenate((np.linspace(0, 0.009, 10), np.linspace(0.01, 0.1, 19),
                                            np.linspace(0.11, 0.8, 70)))

        self.obs_bins_mass = np.concatenate(
          (np.array([0, 1]), np.linspace(1.8, 9.8, 41), np.linspace(10, 14.5, 10),
           np.linspace(15, 19, 5), np.linspace(20, 60, 9)))

        # Lower limit set by zcut; upper limit set by branch tagging (must be < 0.5)
        self.obs_bins_zg = np.concatenate(
          (np.array([0, 0.1]), np.linspace(0.11, 0.5, 40), np.array([0.50001])))

        self.obs_bins_theta_g = np.linspace(0., 1.1, 56)

        self.observable_list = config['process_observables']
        self.obs_settings = {}
        self.obs_grooming_settings = {}
        self.obs_names = {}
        for observable in self.observable_list:

            obs_config_dict = config[observable]
            obs_config_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]

            obs_subconfig_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
            print("obs_subconfig_list", obs_subconfig_list)
            self.obs_settings[observable] = self.utils.obs_settings(observable, obs_config_dict, obs_subconfig_list)
            print("self.obs_settings[observable]", self.obs_settings[observable])
            self.obs_grooming_settings[observable] = self.utils.grooming_settings(obs_config_dict)
            print("self.obs_grooming_settings[observable]", self.obs_grooming_settings[observable])

            self.obs_names[observable] = obs_config_dict["common_settings"]["xtitle"]

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def pythia_quark_gluon(self, args):

        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        # Initialize response histograms
        self.initialize_hist()

        print('user seed for pythia', self.user_seed) #TODO: what does this do?? it doesn't work...
#        print('user seed for pythia', self.user_seed)
        mycfg = ['Random:setSeed=on', 'Random:seed={}'.format(self.user_seed)]
        mycfg.append('HadronLevel:all=off')
        print("charmdecays value", self.charmdecaysOFF)
        if (self.charmdecaysOFF == True and self.replaceKPpairs == False):
            print("charm decays turning OFF")
            # Mesons
            mycfg.append('411:mayDecay = no')    # D+
            mycfg.append('421:mayDecay = no')    # D0
            mycfg.append('10411:mayDecay = no')  # D*0(2400)+
            mycfg.append('10421:mayDecay = no')  # D*0(2400)0
            mycfg.append('413:mayDecay = no')    # D*(2010)+
            mycfg.append('423:mayDecay = no')    # D*(2007)0
            mycfg.append('10413:mayDecay = no')  # D1(2420)+
            mycfg.append('10423:mayDecay = no')  # D1(2420)0
            mycfg.append('20413:mayDecay = no')  # D1(H)+
            mycfg.append('20423:mayDecay = no')  # D1(2430)0
            mycfg.append('415:mayDecay = no')    # D*2(2460)0
            mycfg.append('425:mayDecay = no')    # D*2(2460)0
            mycfg.append('431:mayDecay = no')    # D+s
            mycfg.append('10431:mayDecay = no')  # D*s0(2317)+
            mycfg.append('433:mayDecay = no')    # D*s+
            mycfg.append('10433:mayDecay = no')  # Ds1(2536)+
            mycfg.append('20433:mayDecay = no')  # Ds1(2460)+
            mycfg.append('435:mayDecay = no')    # D*s2(2573)+

            # Baryons
            mycfg.append('4122:mayDecay = no')   # Lc+
            mycfg.append('4222:mayDecay = no')   # Sigmac++
            mycfg.append('4212:mayDecay = no')   # Sigmac+
            mycfg.append('4112:mayDecay = no')   # Sigmac0
            mycfg.append('4224:mayDecay = no')   # Sigma*c++
            mycfg.append('4214:mayDecay = no')   # Sigma*c+
            mycfg.append('4114:mayDecay = no')   # Sigma*c0
            mycfg.append('4232:mayDecay = no')   # Xi+c
            mycfg.append('4132:mayDecay = no')   # Xi0c
            mycfg.append('4322:mayDecay = no')   # Xiprimec+
            mycfg.append('4312:mayDecay = no')   # Xiprimec0
            mycfg.append('4324:mayDecay = no')   # Xi*c+
            mycfg.append('4314:mayDecay = no')   # Xi*c0
            mycfg.append('4332:mayDecay = no')   # Omega0c
            mycfg.append('4334:mayDecay = no')   # Omega*c0
            mycfg.append('4412:mayDecay = no')
            mycfg.append('4422:mayDecay = no')
            mycfg.append('4414:mayDecay = no')
            mycfg.append('4424:mayDecay = no')
            mycfg.append('4432:mayDecay = no')
            mycfg.append('4434:mayDecay = no')
            mycfg.append('4444:mayDecay = no')

        if (self.initscat == 1): #if (self.hardccbar):
            mycfg.append('HardQCD:all = off')
            mycfg.append('HardQCD:hardccbar = on')

        elif (self.initscat == 2): #if (self.gg2ccbar):
            mycfg.append('HardQCD:all = off')
            mycfg.append('HardQCD:gg2ccbar = on')

        elif (self.initscat == 3): # just D0->Kpi
            #mycfg.append('HardQCD:all = off')
            #mycfg.append('HardQCD:hardccbar = on')

            # Uncomment these lines for Z+jet
            mycfg.append('HardQCD:all = off')
            mycfg.append('WeakBosonAndParton:qqbar2gmZg = on')  # q qbar → gamma^*/Z^0 g
            mycfg.append('WeakBosonAndParton:qg2gmZq = on')     # q g → gamma^*/Z^0 q

            mycfg.append('421:onMode = off')
            mycfg.append('421:onIfMatch = 321 211')

        elif (self.initscat == 4): # hard->bbar with D0 -> (only) Kpi
            mycfg.append('HardQCD:all = off')
            mycfg.append('HardQCD:hardbbbar = on')

            mycfg.append('421:onMode = off')
            mycfg.append('421:onIfMatch = 321 211')

        if (self.phimeson):
            print("turning phi's OFF")
            mycfg.append('333:mayDecay = no')
            # mycfg.append('100333:mayDecay = no')
            # mycfg.append('337:mayDecay = no')

        if (self.replaceKPpairs):
            if (not (self.Dstar or self.D0wDstar or self.difNorm)):
                print("turning D*'s OFF")
                mycfg.append('10411:mayDecay = no')
                mycfg.append('10421:mayDecay = no')
                mycfg.append('413:mayDecay = no')
                mycfg.append('423:mayDecay = no')
                mycfg.append('10413:mayDecay = no')
                mycfg.append('10423:mayDecay = no')
                mycfg.append('20413:mayDecay = no')
                mycfg.append('20423:mayDecay = no')
                mycfg.append('415:mayDecay = no')
                mycfg.append('425:mayDecay = no')
                mycfg.append('431:mayDecay = no')
                mycfg.append('10431:mayDecay = no')
                mycfg.append('433:mayDecay = no')
                mycfg.append('10433:mayDecay = no')
                mycfg.append('20433:mayDecay = no')
                mycfg.append('435:mayDecay = no')

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        # -------------------------------
        # Setting MPIs and ISRs
        print('Will run no MPI:', self.noMPI)
        print('Will run no ISR:', self.noISR)
        setattr(args, "py_noMPI", self.noMPI)
        setattr(args, "py_noISR", self.noISR)
        # -------------------------------

        self.pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
        # print("----------------- PARTICLE DATA INFO HERE -----------------")
        # pythia.particleData.listAll()
        # print("----------------- PARTICLE DATA INFO END -----------------")

        self.init_jet_tools()
        self.calculate_events(self.pythia)
        self.pythia.stat()
        print()

        self.scale_print_final_info(self.pythia)

        outf.Write()
        outf.Close()

        self.save_output_objects()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):

        self.make_durations = False  # Can slow down batch processing, for tests only
        if self.make_durations:
            self.jade_durations = []
            self.wta_durations = []

        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)
        self.hD0Nevents = ROOT.TH1I("hD0Nevents", "Total Number of D0 events (unscaled)", 2, -0.5, 1.5)
        self.hD0KpiNevents = ROOT.TH1I("hD0KpiNevents", "Number of D0->Kpi events (unscaled)", 2, -0.5, 1.5)
        self.hD0KpiNjets = ROOT.TH1I("hD0KpiNjets", "Number of D0->Kpi jets (unscaled)", 2, -0.5, 1.5) #accidentally called "hD0KpiNehD0KpiNjetsvents"
        self.hDstarNjets = ROOT.TH1I("hDstarNjets", "Number of D* jets (unscaled)", 2, -0.5, 1.5)
        self.hsoftpionpT = ROOT.TH1D("hsoftpionpT", "pT of soft pion from D*", 50, 0, 50)
        self.hDeltaR = ROOT.TH1F("hDeltaR", 'Delta R between jet and each parent', 40, 0, 0.4)

        if self.phimeson:
            self.hphiNevents = ROOT.TH1I("hphiNevents", "Total Number of phi events (unscaled)", 2, -0.5, 1.5)
            self.hphiNjets = ROOT.TH1I("hphiNjets", "Number of phi jets (unscaled)", 2, -0.5, 1.5)

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            R_label = str(jetR).replace('.', '') + 'Scaled'

            for observable in self.observable_list:

                if observable not in ["mass", "zg", "theta_g"]:
                    raise ValueError("Observable %s is not implemented in this script" % observable)

                obs_name = self.obs_names[observable]
                obs_bins = getattr(self, "obs_bins_" + observable)
                # Use more finely binned pT bins for TH2s than for the RMs
                pt_bins = array('d', list(range(0, 201, 1)))
                #rapi_bins = np.linspace(-5,5,201)

                #dim = 4
                #nbins  = [len(pt_bins)-1, len(pt_bins)-1, len(rapi_bins)-1, 50]
                #min_li = [pt_bins[0],     pt_bins[0],      rapi_bins[0],      obs_bins[0]]
                #max_li = [pt_bins[-1],    pt_bins[-1],     rapi_bins[-1],     obs_bins[-1]]

                #nbins = (nbins)
                #xmin = (min_li)
                #xmax = (max_li)

                #nbins_array = array('i', nbins)
                #xmin_array = array('d', xmin)
                #xmax_array = array('d', xmax)

                # Loop over subobservable (alpha value)
                for i in range(len(self.obs_settings[observable])):

                    obs_setting = self.obs_settings[observable][i]
                    grooming_setting = self.obs_grooming_settings[observable][i]
                    obs_label = self.utils.obs_label(obs_setting, grooming_setting)
                    print("all the settings", obs_setting, grooming_setting, obs_label)

                    partontypeslist = ["charm"] #, "light", "gluon", "inclusive"]
                    #if (self.initscat == 4):
                    #    partontypeslist.append("beauty")

                    for parton_type in partontypeslist:

                        title = [ '#it{p}_{T}^{ch jet}', '#it{p}_{T}^{D^{0}}', obs_name]

                        # make TH3D for observable
                        name = ('h3D_%s_JetPt_%s_R%s_%s' % (observable, parton_type, jetR, obs_label)) if \
                            len(obs_label) else ('h3D_%s_JetPt_%s_R%s' % (observable, parton_type, jetR))
                        h3D = ROOT.TH3D(name, name, len(pt_bins)-1, pt_bins, len(pt_bins)-1, pt_bins, len(obs_bins)-1, obs_bins)
                        h3D.Sumw2()
                        h3D.GetXaxis().SetTitle(title[0])
                        h3D.GetYaxis().SetTitle(title[1])
                        h3D.GetZaxis().SetTitle(title[2])
                        '''
                        for i in range(0, 3):
                            hsparse.GetAxis(i).SetTitle(title[i])
                            if i == 0 or i == 1:
                                hsparse.SetBinEdges(i, pt_bins)
                            if i == 2:
                                hsparse.SetBinEdges(i, rapi_bins)
                            if i == 3:
                                hsparse.SetBinEdges(i, obs_bins)
                        '''
                        setattr(self, name, h3D)
                        getattr(self, hist_list_name).append(h3D)

                        # jetflav IRC-safe algorithms
                        for algo_name in ["JADE", "WTA", "IFN", "CMP", "GHS"]:
                            name = ('h3D_%s_%s_JetPt_%s_R%s_%s' % (observable, algo_name, parton_type, jetR, obs_label)) if \
                                len(obs_label) else ('h3D_%s_%s_JetPt_%s_R%s' % (observable, algo_name, parton_type, jetR))
                            h3D = ROOT.TH3D(name, name, len(pt_bins)-1, pt_bins, len(pt_bins)-1, pt_bins, len(obs_bins)-1, obs_bins)
                            h3D.Sumw2()
                            h3D.GetXaxis().SetTitle(title[0])
                            h3D.GetYaxis().SetTitle(title[1])
                            h3D.GetZaxis().SetTitle(title[2])
                            setattr(self, name, h3D)
                            getattr(self, hist_list_name).append(h3D)

    #---------------------------------------------------------------
    # Initiate jet defs, selectors, and sd (if required)
    #---------------------------------------------------------------
    def init_jet_tools(self):

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')

            # set up our jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            setattr(self, "jet_def_R%s" % jetR_str, jet_def)

        #print('eta range for particles after hadronization set to', self.min_eta_hadron, "< eta <", self.max_eta_hadron)
        parts_selector_h = fj.SelectorAbsEtaMin(self.min_eta_hadron)
        parts_selector_h &= fj.SelectorAbsEtaMax(self.max_eta_hadron)
        parts_selector_h &= fj.SelectorPtMin(0.10)
        setattr(self, "parts_selector_h", parts_selector_h)
        parts_selector_ch = parts_selector_h
        setattr(self, "parts_selector_ch", parts_selector_ch)
        print("Particle selector is:", parts_selector_h.description())

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')

            self.jetptcut = 5.0
            jet_selector = fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            if self.min_eta_hadron != 0:
                jet_selector &= fj.SelectorAbsEtaMin(self.min_eta_hadron + jetR)
            jet_selector &= fj.SelectorPtMin(self.jetptcut)
            #jet_selector = fj.SelectorPtMin(0.) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)
            print("For R=%.1f, jet selector is:" % jetR, jet_selector.description())

            count1 = 0  # Number of partonic parents which match to >1 ch-jets
            setattr(self, "count1_R%s" % jetR_str, count1)
            count2 = 0  # Number of partonic parents which match to zero ch-jets
            setattr(self, "count2_R%s" % jetR_str, count2)

    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def calculate_events(self, pythia):

        iev = 0  # Event loop count

        self.parton_counter = 0

        while iev < self.nev:
            if not pythia.next():
                continue

            self.event = pythia.event

            # Check if the event contains desired parton, else continue
            desired_pid = [4] #, 5] # charm, bottom quark
            desired_parton_found = False
            for parton in pythia.event:
                if parton.id() in desired_pid:
                    if (self.min_eta_hadron - 1) <= abs(parton.eta()) <= (self.max_eta_hadron + 1):
                        desired_parton_found = True
                        break
            if not desired_parton_found:
                self.parton_counter += 1
                continue
            #is_desired_parton = [abs(particle.id()) in desired_pid for particle in self.event]
            #if True not in is_desired_parton:
            #    continue

            # print(self.event) # to print out a table of the event information
            fs_parton_5 = fj.PseudoJet(pythia.event[5].px(), pythia.event[5].py(), pythia.event[5].pz(), pythia.event[5].e())
            fs_parton_6 = fj.PseudoJet(pythia.event[6].px(), pythia.event[6].py(), pythia.event[6].pz(), pythia.event[6].e())
            self.parents = [fs_parton_5, fs_parton_6] # parent partons in dijet

            # Save PDG code of the parent partons
            self.parent_ids = [pythia.event[5].id(), pythia.event[6].id()]

            # parton level
            parts_pythia_p = pythiafjext.vectorize_select(pythia, vector[int]([pythiafjext.kFinal]), 0, True)

            # Get hadron-level event
            require_pid = 421      # D0 particle ID
            daughters_size = -1    # require certain number of daughters (disabled = -1)
            if self.Dstar:
                require_pid = 413
                dauthers_size = 2
            elif self.replaceKPpairs:  # D0 or D0-with-D*
                dauthers_size = 2
            elif self.phimeson:
                require_pid = 333  # phi meson particle ID

            # If successful, returns index of the satisfiying particle in the event, else -1
            satisfier_ip = pythiafjext.update_hadronization(pythia, require_pid, daughters_size)
            if satisfier_ip == -1:
                continue

            # full-hadron level
            if ( self.replaceKPpairs == False ):
                parts_pythia_h = pythiafjext.vectorize_select(pythia, vector[int]([pythiafjext.kFinal]), 0, True)
                # print("There are ", len(old_pythia_hch), "in oph, and ", len(phis_pythia_hch), " in phis", len(parts_pythia_hch), ">
            else: #replace D0->Kpi
                if ( self.softpion_action != 1):
                    parts_pythia_h = pythiafjext.vectorize_select_replaceD0(pythia, vector[int]([pythiafjext.kFinal]), 0, True)
                else:
                    parts_pythia_h = pythiafjext.vectorize_select_replaceD0(pythia, vector[int]([pythiafjext.kFinal]), 0, True, True)

            # print("!! pythia hadron (before vectorization) event size is ", pythia.event.size())
            # eventcounter = 0
            # for event in pythia.event:
            #     # if event.id() == 111 or event.id() == 211 or event.id() == -211: #pi0 or pi+ or pi-
            #         # print(eventcounter, "pion with event id", event.id())
            #     eventcounter+=1

            #testing
            # parts_pythia_hch_noreplace = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)
            # parts_pythia_hch_replaced = pythiafjext.vectorize_select_replaceD0(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # charged-hadron level
            if ( self.replaceKPpairs == False ):
                if ( self.phimeson ):
                    old_pythia_hch = pythiafjext.vectorize_select(pythia, vector[int]([pythiafjext.kFinal, pythiafjext.kCharged]), 0, True)
                    phis_pythia_hch = pythiafjext.vectorize_select(pythia, vector[int]([pythiafjext.kFinal, pythiafjext.kPhi]), 0, True)
                    parts_pythia_hch = pythiafjext.add_vectors(old_pythia_hch, phis_pythia_hch)
                    # print("There are ", len(old_pythia_hch), "in oph, and ", len(phis_pythia_hch), " in phis", len(parts_pythia_hch), " in parts")
                else:
                    parts_pythia_hch = pythiafjext.vectorize_select(pythia, vector[int]([pythiafjext.kFinal, pythiafjext.kCharged]), 0, True)
            else: #replace D0->Kpi
                if ( self.softpion_action != 1):
                    parts_pythia_hch = pythiafjext.vectorize_select_replaceD0(pythia, vector[int]([pythiafjext.kFinal, pythiafjext.kCharged]), 0, True)
                else:
                    parts_pythia_hch = pythiafjext.vectorize_select_replaceD0(pythia, vector[int]([pythiafjext.kFinal, pythiafjext.kCharged]), 0, True, True)
            # print("Size of 1 vector", len(parts_pythia_hch_noreplace))
            # print("Size of 2 vector", len(parts_pythia_hch_replaced))
            # print("Size of new vector", len(parts_pythia_hch))

            # look at events in charged hadron??
            # print("!! pythia hadron (after vectorization) event size is ", pythia.event.size())
            #TODO: move this block above choosing final state charged particles??
            particlecounter = 0
            D0found = False
            D0Kpidecayfound = False
            phifound = False
            self.DstarKpipidecayfound = False
            #for ip in range(satisifer_ip, len(self.event)):
            particle = self.event[satisfier_ip]
            if abs(particle.id()) == 421 and self.min_eta_hadron <= abs(particle.eta()) <= self.max_eta_hadron: #D0
                D0found = True
                decayChannel = self.checkDecayChannel(particle, self.event)
                if decayChannel == EMesonDecayChannel.kDecayD0toKpi:
                    #print("D0 eta:", particle.eta(), "// D0 pT:",
                    #    math.sqrt(particle.px()*particle.px() + particle.py()*particle.py()))
                    D0Kpidecayfound = True
                if decayChannel == EMesonDecayChannel.kDecayDStartoKpipi:
                    self.DstarKpipidecayfound = True
            elif abs(particle.id()) == 413: # D*
                self.DstarKpipidecayfound = True
            elif abs(particle.id()) == 333: # phi
                phifound = True

            #if D0->Kpi found, count the events; if not, check that length of charged final state hadrons vector is 0
            if (D0Kpidecayfound):
                self.hD0KpiNevents.Fill(0)
            if (D0found):
                self.hD0Nevents.Fill(0)
            if (self.phimeson and phifound):
                self.hphiNevents.Fill(0)

            # Some "accepted" events don't survive hadronization step -- keep track here
            self.hNevents.Fill(0)
            self.find_jets_fill_histograms(parts_pythia_h, parts_pythia_hch, iev, D0Kpidecayfound)

            if (iev%100 == 0):
                print("Event", iev)
            # print("Event", iev)

            iev += 1

    #---------------------------------------------------------------
    # Find primordial parent
    #---------------------------------------------------------------
    def primordial_parent(self,p):
        parent1 = parent2 = -10
        while p > 6:
            parent1 = self.event[p].mother1()
            parent2 = self.event[p].mother2()
            if parent1 != parent2:
                p = max(parent1,parent2)
            else:
                p = parent1
        return p

    # trk_thrd default set 0, meaning all tracks would pass
    def checkIfPartInJetConst(self, jet_const_arr, pythia_particle_index, trk_thrd=0):
        in_jet = False
        for c in jet_const_arr:
            # print("jet const user index", c.user_index(), pythiafjext.getPythia8Particle(c).name())
            if (c.user_index() == pythia_particle_index and c.pt() >= trk_thrd):
                in_jet = True
                # print("ifpartinjet", c.user_index(), pythia_particle_index)
                break
        return in_jet

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms
    #---------------------------------------------------------------
    def find_jets_fill_histograms(self, parts_pythia_h, parts_pythia_hch, iev, D0Kpidecayfound):

        # Don't waste time if there are no D0 mesons
        if not D0Kpidecayfound:
            return

        parts_selector_h = getattr(self, "parts_selector_h")
        parts_selector_ch = getattr(self, "parts_selector_ch")

        # Loop over jet radii
        for jetR in self.jetR_list:

            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
            jet_def = getattr(self, "jet_def_R%s" % jetR_str)

            count1 = getattr(self, "count1_R%s" % jetR_str)
            count2 = getattr(self, "count2_R%s" % jetR_str)

            # Get the jets at different levels
            #jets_p  = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p  ))) # parton level
            #jets_h  = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_h  ))) # full hadron level
            if (not self.replaceKPpairs and not self.phimeson):
                jets_h = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_h)))
                jets_ch = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_hch))) # charged hadron level
            else:
                jets_h = fj.sorted_by_pt(jet_selector(jet_def(parts_selector_h(parts_pythia_h))))
                jets_ch = fj.sorted_by_pt(jet_selector(jet_def(parts_selector_ch(parts_pythia_hch)))) # charged hadron level

            # print("!! length of jets_ch", len(jets_ch))

            R_label = str(jetR).replace('.', '') + 'Scaled'

            ''' Matching jet to parent parton
            # Find the charged jet closest to the axis of the original parton
            # Require that the match is within some small angle, and that it is unique
            jet_matching_distance = 0.6  # Match jets with deltaR < jet_matching_distance*jetR
            self.parent0match, self.parent1match = None, None
            anothacounter=0
            # print("LOOPING OVER JETS")
            for i_jch, jch in enumerate(jets_ch):
                # print(i_jch)
                # Do constituent pT cut
#                print("self.min_leading_track_pT", self.min_leading_track_pT)
#                 if self.min_leading_track_pT and not \
#                     self.utils.is_truth_jet_accepted(jch):
# #                   self.utils.is_truth_jet_accepted(jch, self.min_leading_track_pT):
#                     continue
                # print("PARENTS:",self.parents)
                for i_parent, parent in enumerate(self.parents):
                    anothacounter+=1
                    parentmatch_name = "parent%imatch" % i_parent
                    # print("CHECKING PARENT", i_parent)
                    # print("DELTA R TO JET:", jch.delta_R(parent))
                    #plot 
                    self.hDeltaR.Fill(jch.delta_R(parent))
                    if jch.delta_R(parent) < jet_matching_distance * jetR:
                        match = getattr(self, parentmatch_name)
                        # print("MATCH FOR",i_parent,":",match)
                        if not match:
                            setattr(self, parentmatch_name, jch)
                            # print("MATCH SET TO JET WITH pT", jch.pt())
                        else:  # Already found a match
                            # Set flag value so that we know to ignore this one
                            # print("already found a match flagged")
                            setattr(self, parentmatch_name, 0)
                    # print(i_jch, "anothacounter", anothacounter)

            # print("event num", iev)
            # print("Cehckpoint 1")

            # If we have matches, fill histograms
            for i_parent, parent in enumerate(self.parents):
                # print("in nexr loop")
                jet = getattr(self, "parent%imatch" % i_parent)
                # print(jet)
                if not jet:
                    # print("in not jet")
                    if jet == 0: # More than one match -- take note and continue
                        # print("case  1")
                        count1 += 1
                        continue
                    else:  # jet == None
                        # No matches -- take note and continue
                        # print("case  2")
                        count2 += 1
                        continue

                # print("CHECKPOINT 2")

                # One unique match
                # Identify the histograms which need to be filled
#                print("passed not jet")
                parton_id = self.parent_ids[i_parent]
                # print("parton_id is ", parton_id)
                parton_types = []
                if parton_id in self.quark_pdg_ids:
                    # parton_types += ["quark"]
                    if parton_id in self.charm_pdg_ids:
                        parton_types += ["charm"]
                    elif parton_id in self.up_pdg_ids or parton_id in self.down_pdg_ids or parton_id in self.strange_pdg_ids:
                        parton_types += ["light"]
                    elif (parton_id in self.beauty_pdg_ids and self.initscat == 4):
                        parton_types += ["beauty"]
                elif parton_id in self.gluon_pdg_ids:
                    parton_types += ["gluon"]
                if self.phimeson:
                    parton_types += ["inclusive"]

                # If parent parton not identified, skip for now
                if not len(parton_types):
                    continue

                # print(D0Kpidecayfound)
                if D0Kpidecayfound:
                    print("parton types", parton_types)
                '''

            # Loop over jets
            # Because of different definitions, have to do multiple loops
            # First AKT loop; for IFN loop, see below

            # AKT full jets
            for i_jh, jet in enumerate(jets_h):

                # Select for just D0-tagged jets #TODO: check if this D0 goes to kaon pion??
                D0taggedjet = False
                N_D0 = 0
                Dstartaggedjet = False
                if ( self.replaceKPpairs ):
                    # print("There are ", len(jet.constituents()), "constituents.")
                    for c in jet.constituents():
                        constituent_pdg_idabs = pythiafjext.getPythia8Particle(c).idAbs()
                        constituent_pdg_index = c.user_index()
                        if (constituent_pdg_idabs == 421):
                            decayChannel = self.checkDecayChannel(pythiafjext.getPythia8Particle(c), self.event)
                            if (decayChannel == EMesonDecayChannel.kDecayD0toKpi):
                                self.getD0Info(pythiafjext.getPythia8Particle(c))
                                N_D0 += 1
                                D0taggedjet = True
                                #break
                            elif (decayChannel == EMesonDecayChannel.kDecayDStartoKpipi):
                                self.getD0Info(pythiafjext.getPythia8Particle(c))
                                Dstartaggedjet = True

                                # save soft pion info from D* if needed
                                if ( self.softpion_action >= 2 ): #(self.Dstar):
                                    # get the soft pion
                                    if (self.checkD0motherIsDstar(self.D0particleinfo, self.event)):
                                        # print("mother is in fact a Dstar")
                                        softpion_index = self.getSoftPion(self.D0particleinfo, self.event, jet.constituents())
                                        # print("the soft pion index is", softpion_index)
                                        if softpion_index == -1:
                                            self.softpion_particleinfo_psjet = None
                                        else:
                                            self.softpion_particleinfo_psjet = self.getParticleAsPseudojet(self.event[softpion_index])
                                            softpion_pt = self.event[softpion_index].pT()
                                            self.hsoftpionpT.Fill(softpion_pt)
                                        self.D0particleinfo_psjet = self.getParticleAsPseudojet(self.event[constituent_pdg_index])#self.D0particleinfo)

                                break

                    # Skip jets that are not dtagged or dstar tagged
                    if not (D0taggedjet or Dstartaggedjet):
                        #print("continuing......")
                        continue
                    elif N_D0 > 1:
                        print("Found %i D0 particles in this jet, continuing" % N_D0)
                        continue

                    # Select D* jets when required
                    if ( self.difNorm == False ):
                        if ( not self.Dstar and not self.D0wDstar ):
                            if ( not D0taggedjet ): #if not a D0 tagged jet, move to next jet
                                print("Dstar is false, D0wDstar is false, and this is not D0tagged jet")
                                continue
                        if ( self.Dstar and not Dstartaggedjet ): #if only looking at D*s and D* is not tagged, move to next jet
                            # print("Dstar is true and Dstar is not tagged")
                            continue

                    # check if prompt or nonprompt - NOT for the Dstars - TODO: CHECK THIS LATER
                    # if (not self.Dstar):
                    #     self.promptness = self.checkPrompt(pythiafjext.getPythia8Particle(c), self.event)
                    #     print("prompt???", self.promptness)

                phitaggedjet = False
                # print("There are ", len(jet.constituents()), "constituents.")    
                if (self.phimeson):
                    for c in jet.constituents():
                        constituent_pdg_idabs = pythiafjext.getPythia8Particle(c).idAbs()
                        constituent_pdg_index = c.user_index()
                        # print("const index from pythiafjext", pythiafjext.getPythia8Particle(c).index(), constituent_pdg_idabs, constituent_pdg_index)
                        # print("const user_index from pythiafjext", c.user_index())
                        if (constituent_pdg_idabs == 333): #TODO: this is assuming there is only one phi per jet!
                            print("phi jet!")
                            self.getD0Info(pythiafjext.getPythia8Particle(c)) #naming here is bad but just wanted to reuse the variable
                            phitaggedjet = True
                            break

                    # move on if this jet doesn't have a phi meson
                    if ( not phitaggedjet ):
                        # print("Not a phi jet")
                        continue

                # count the number of D0-tagged jets
                if (D0taggedjet): #D0Kpidecayfound):
                    self.hD0KpiNjets.Fill(0)
                if (Dstartaggedjet): #self.DstarKpipidecayfound):
                    self.hDstarNjets.Fill(0)
                if (self.phimeson and phitaggedjet):
                    self.hphiNjets.Fill(0)

                if not (jet.has_constituents() or jet.has_structure()):
                    continue

                ###############################################################
                # Check for tagging using various algorithms
                ##############################################################

                ##############################################################
                # JADE
                start_time_jade = 0
                if self.make_durations:
                    start_time_jade = time.time()
                jade_tagged = False
                JADE_SD_BETA = 1;  JADE_SD_ZCUT = 0.1
                reclusterer_jade = fj.JadePlugin()
                jet_def_jade = fj.JetDefinition(reclusterer_jade)
                lg_jade = fj.contrib.LundGenerator(jet_def_jade)
                jet_gr_jade = fj.PseudoJet()  # initialize empty jet
                for ld in lg_jade.result(jet):  # Vector of LundDeclustering (reclustered with JADE)
                    if ld.z() > JADE_SD_ZCUT * (ld.Delta() / jetR) ** JADE_SD_BETA:  # SD condition
                        jet_gr_jade = ld.pair()
                        break
                if jet_gr_jade.has_constituents() and jet_gr_jade.has_structure():
                    # Look for a D0 inside the jet
                    for constit in jet_gr_jade.constituents():
                        pid_abs = pythiafjext.getPythia8Particle(constit).idAbs()
                        pid_abs_str = str(pid_abs)
                        if pid_abs == 421:
                            #n_HF_meson_found += 1
                            jade_tagged = True
                            continue
                        elif (len(pid_abs_str) >= 3 and pid_abs_str[-3] == '4') or \
                            (len(pid_abs_str) >= 4 and pid_abs_str[-4] == '4'):
                            # Jet has some non-D0 charm -- untag
                            jade_tagged = False
                            break
                #if n_HF_meson_found == 1:
                #   jade_tagged = True
                if self.make_durations:
                    self.jade_durations.append(time.time() - start_time_jade)

                ##############################################################
                # WTA RECLUSTERING
                start_time_wta = 0
                if self.make_durations:
                    start_time_wta = time.time()
                wta_tagged = False
                # fastjet::max_allowable_R == 1000.0
                jet_def_wta = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
                jet_def_wta.set_recombination_scheme(fj.WTA_pt_scheme)
                recluster_wta = fj.contrib.Recluster(jet_def_wta)
                jet_wta = recluster_wta.result(jet)
                if jet_wta.has_constituents():
                    # Loop through constituents to find the one aligned with WTA axis
                    for constit in jet_wta.constituents():
                        if constit.delta_R(jet_wta) < 1e-8:
                            # Particle found
                            if pythiafjext.getPythia8Particle(constit).idAbs() == 421:
                                wta_tagged = True
                            break
                if self.make_durations:
                    self.wta_durations.append(time.time() - start_time_wta)

                ##############################################################
                # Interleaved Flavor Neutralization (IFN)

                ## Recluster AKT jet using Flavor Recombiner (uses FlavInfo class)
                #for c in jet.constituents():
                #    pdg_id = pythiafjext.getPythia8Particle(c).id()
                #    c.set_user_info(FlavHistory(pdg_id))
                #jet_def_IFN_base = fj.JetDefinition(fj.antikt_algorithm, 1000.0)
                #flav_recombiner = fj.contrib.FlavRecombiner()
                #jet_def_IFN_base.set_recombiner(flav_recombiner)
                #recluster_IFN_base = fj.contrib.Recluster(jet_def_IFN_base)
                #jet_IFN_base = recluster_IFN_base.result(jet)

                #################################################################

                # Fill histograms
                for observable in self.observable_list:
                    #print("len(self.obs_settings[observable])", len(self.obs_settings[observable]))
                    for i in range(len(self.obs_settings[observable])):

                        obs_setting = self.obs_settings[observable][i]
                        grooming_setting = self.obs_grooming_settings[observable][i]
                        obs_label = self.utils.obs_label(obs_setting, grooming_setting)

                        # Groom jet, if applicable
                        jet_groomed_lund = None
                        if grooming_setting:
                            if 'sd' not in grooming_setting:
                                raise NotImplementedError("Only SD grooming is implemented")
                            zcut = grooming_setting['sd'][0]
                            beta = grooming_setting['sd'][1]
                            lg = fj.contrib.LundGenerator(self.reclustering_algorithm)
                            for ld in lg.result(jet):  # Vector of LundDeclustering
                                if ld.z() > zcut * (ld.Delta() / jetR) ** beta:  # SD condition
                                    jet_groomed_lund = ld
                                    break
                            if not jet_groomed_lund:
                                continue

                        # Apply cut on leading track pT
                        if self.leading_parton_pt_cut:
                            leading_parton = fj.sorted_by_pt(jet.constituents())[0]
                            if (leading_parton.pt() < self.leading_parton_pt_cut):
                                continue

                        # skip filling the pair level information if necessary
                        if (self.difNorm):
                            if ( not self.Dstar and not self.D0wDstar ):
                                if ( Dstartaggedjet ):
                                    continue
                            elif ( self.Dstar ):
                                if ( D0taggedjet ):
                                    continue

                        if (self.softpion_action >= 2 and Dstartaggedjet):
                            if (softpion_index == -1): #skip because soft pion is not in the jet!
                                continue

                        obs = self.calculate_observable(
                            observable, jet, jet_groomed_lund, jetR, obs_setting,
                            grooming_setting, obs_label, jet.pt())

                        for parton_type in ["charm"]: # parton_types:
                            #fill parton hnsparse info
                            if (self.replaceKPpairs or self.phimeson): # phimeson has bad naming convention but is properly filled here
                                D0_px = self.D0particleinfo.px()
                                D0_py = self.D0particleinfo.py()
                                D0_pt = math.sqrt(D0_px * D0_px + D0_py * D0_py)
                            else:
                                D0_pt = -1

                            # Traditional anti-kT jet
                            getattr(self, ('h3D_%s_JetPt_%s_R%s_%s' % (observable, parton_type, jetR, obs_label)) if \
                                len(obs_label) else ('h3D_%s_JetPt_%s_R%s' % (observable, parton_type, jetR))).Fill(
                                jet.pt(), D0_pt, obs)

                            # Custom tagged jets
                            if jade_tagged:
                                getattr(self, ('h3D_%s_JADE_JetPt_%s_R%s_%s' % (observable, parton_type, jetR, obs_label)) if \
                                    len(obs_label) else ('h3D_%s_JADE_JetPt_%s_R%s' % (observable, parton_type, jetR))).Fill(
                                    jet.pt(), D0_pt, obs)

                            if wta_tagged:
                                getattr(self, ('h3D_%s_WTA_JetPt_%s_R%s_%s' % (observable, parton_type, jetR, obs_label)) if \
                                    len(obs_label) else ('h3D_%s_WTA_JetPt_%s_R%s' % (observable, parton_type, jetR))).Fill(
                                    jet.pt(), D0_pt, obs)

            # jetflav IRC-safe jets
            if self.replaceKPpairs or self.phimeson:
                pdg_ids = [pythiafjext.getPythia8Particle(p).id() for p in parts_pythia_h]
                jet_def_base = fj.JetDefinition(fj.antikt_algorithm, jetR)
                flav_recombiner = fj.contrib.FlavRecombiner()
                jet_def_base.set_recombiner(flav_recombiner)

                # IFN jets
                IFN_alpha = 2;  IFN_omega = 3 - IFN_alpha;  # IFN constants
                flav_summation = fj.contrib.FlavRecombiner.net
                IFN_plugin = fj.contrib.IFNPlugin(jet_def_base, IFN_alpha, IFN_omega, flav_summation)
                jet_def_IFN = fj.JetDefinition(IFN_plugin)
                for ip, p in enumerate(parts_pythia_h):  # Reset user info
                    p.set_user_info(fj.contrib.FlavHistory(pdg_ids[ip]))
                jets_h_IFN = fj.sorted_by_pt(jet_selector(jet_def_IFN(parts_selector_h(parts_pythia_h))))

                for i_jh, jet in enumerate(jets_h_IFN):
                    self.get_flav_fill_histograms(jet, jetR, "IFN")

                # GHS jets
                GHS_alpha = 1;  GHS_omega = 2;  # GHS constants
                for ip, p in enumerate(parts_pythia_h):  # Reset user info
                    p.set_user_info(fj.contrib.FlavHistory(pdg_ids[ip]))
                jets_h_GHS_base = jet_selector(jet_def_base(parts_selector_h(parts_pythia_h)))
                jets_h_GHS = fj.sorted_by_pt(fj.contrib.run_GHS(jets_h_GHS_base, self.jetptcut, GHS_alpha, GHS_omega, flav_recombiner))

                for i_jh, jet in enumerate(jets_h_GHS):
                    self.get_flav_fill_histograms(jet, jetR, "GHS")

                # CMP jets
                CMP_a = 0.1  # 'a' parameter in:   kappa_ij = 1/a * (kT_i^2 + kT_j^2) / (2*kT_max^2)
                CMP_corr = fj.CMPPlugin.CorrectionType.OverAllCoshyCosPhi_a2  # IRC-safe correction
                CMP_clust = fj.CMPPlugin.ClusteringType.DynamicKtMax          # Dynamic def of ktmax
                CMP_plugin = fj.CMPPlugin(jetR, CMP_a, CMP_corr, CMP_clust)
                jet_def_CMP = fj.JetDefinition(CMP_plugin)
                flav_recombiner = fj.contrib.FlavRecombiner()
                jet_def_CMP.set_recombiner(flav_recombiner)
                for ip, p in enumerate(parts_pythia_h):  # Reset user info
                    p.set_user_info(fj.contrib.FlavHistory(pdg_ids[ip]))
                jets_h_CMP = fj.sorted_by_pt(jet_selector(jet_def_CMP(parts_selector_h(parts_pythia_h))))

                for i_jh, jet in enumerate(jets_h_CMP):
                    self.get_flav_fill_histograms(jet, jetR, "CMP")

            setattr(self, "count1_R%s" % jetR_str, count1)
            setattr(self, "count2_R%s" % jetR_str, count2)

    #---------------------------------------------------------------
    # Get IRC-safe jet flavor and fill appropriate histograms
    #---------------------------------------------------------------
    def get_flav_fill_histograms(self, jet, jetR, algo_name):

        flav = fj.contrib.FlavHistory.current_flavour_of(jet).description()
        charm_tagged = 'c' in flav
        if not charm_tagged:
            return

        # Save D0 info -- TODO: check for D*
        D0meson = None
        pdg_ids = []
        if ( self.replaceKPpairs ):
            # print("There are ", len(jet.constituents()), "constituents.")
            for c in jet.constituents():
                flav_info = fj.contrib.FlavHistory.current_flavour_of(c)
                pdg_idabs = abs(flav_info.pdg_code())
                pdg_ids.append(pdg_idabs)
                if (pdg_idabs == 421):
                    D0meson = c
            if not D0meson:  # There is charm but not D0
                return

        # Fill histograms
        for observable in self.observable_list:
            #print("len(self.obs_settings[observable])", len(self.obs_settings[observable]))
            for i in range(len(self.obs_settings[observable])):

                obs_setting = self.obs_settings[observable][i]
                grooming_setting = self.obs_grooming_settings[observable][i]
                obs_label = self.utils.obs_label(obs_setting, grooming_setting)

                # Groom jet, if applicable
                jet_groomed_lund = None
                if grooming_setting:
                    if 'sd' not in grooming_setting:
                        raise NotImplementedError("Only SD grooming is implemented")
                    zcut = grooming_setting['sd'][0]
                    beta = grooming_setting['sd'][1]
                    lg = fj.contrib.LundGenerator(self.reclustering_algorithm)
                    for ld in lg.result(jet):  # Vector of LundDeclustering
                        if ld.z() > zcut * (ld.Delta() / jetR) ** beta:  # SD condition
                            jet_groomed_lund = ld
                            break
                    if not jet_groomed_lund:
                        return

                # Apply cut on leading track pT
                if self.leading_parton_pt_cut:
                    leading_parton = fj.sorted_by_pt(jet.constituents())[0]
                    if (leading_parton.pt() < self.leading_parton_pt_cut):
                        return

                obs = self.calculate_observable(
                    observable, jet, jet_groomed_lund, jetR, obs_setting,
                    grooming_setting, obs_label, jet.pt())

                for parton_type in ["charm"]: # parton_types:
                    #fill parton hnsparse info
                    if (self.replaceKPpairs or self.phimeson): # phimeson has bad naming convention but is properly filled here
                        D0_px = D0meson.px()
                        D0_py = D0meson.py()
                        D0_pt = math.sqrt(D0_px * D0_px + D0_py * D0_py)
                    else:
                        D0_pt = -1

                    # Fill appropriate jet histograms
                    getattr(self, ('h3D_%s_%s_JetPt_%s_R%s_%s' % (observable, algo_name, parton_type, jetR, obs_label)) if \
                        len(obs_label) else ('h3D_%s_%s_JetPt_%s_R%s' % (observable, algo_name, parton_type, jetR))).Fill(
                        jet.pt(), D0_pt, obs)

    #---------------------------------------------------------------
    # Calculate the observable given a jet
    #---------------------------------------------------------------
    def calculate_observable(self, observable, jet, jet_groomed_lund,
        jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed):

        # Jet invariant mass
        if observable == "mass":

            if grooming_setting:
                j_groomed = jet_groomed_lund.pair()
                if not j_groomed.has_constituents():
                    # Untagged jet -- record underflow value
                    return -1
                else:
                    return j_groomed.m()

            return jet.m()

        # Groomed jet radius
        elif observable == "theta_g":

            if not grooming_setting:
                raise ValueError("Cannot calculate theta_g without grooming, check config")
            j_groomed = jet_groomed_lund.pair()
            if not j_groomed.has_constituents():
                # Untagged jet -- record underflow value
                return -1
            return jet_groomed_lund.Delta() / jetR

        # Groomed jet momentum splitting fraction
        elif observable == "zg":

            if not grooming_setting:
                raise ValueError("Cannot calculate zg without grooming, check config")
            j_groomed = jet_groomed_lund.pair()
            if not j_groomed.has_constituents():
                # Untagged jet -- record underflow value
                return -1
            return jet_groomed_lund.z()

        # Should not be any other observable
        raise ValueError("Observable %s not implemented" % observable)


    def checkDecayChannel(self, particle, event): #(part, mcArray): # what type is part

        if(not event):
            return EMesonDecayChannel.kUnknownDecay

        decay = EMesonDecayChannel.kUnknownDecay

        absPdgPart = particle.idAbs()

        if(len(particle.daughterList()) == 2):
            d1_index = particle.daughterList()[0] #don't use daughter1() and daughter(2)
            d2_index = particle.daughterList()[1]
            d1 = event[d1_index]
            d2 = event[d2_index]

            if(not d1 or not d2):
                return decay

            # print("checkpoint 3")


            absPdg1 = d1.idAbs()
            absPdg2 = d2.idAbs()

            if(absPdgPart == 421):  # D0 -> K pi
                if((absPdg1 == 211 and absPdg2 == 321) or (absPdg1 == 321 and absPdg2 == 211)): # pi K or K pi - QUESTION: does this account for k and pi being opposite signs?
                    decay = EMesonDecayChannel.kDecayD0toKpi

            # TODO: can insert if (self.Dstar) later

            # Look at D0's mother particles
            # print("current particle ID is", absPdgPart)
            mother_indices = particle.motherList()
            if (len(mother_indices) != 1):
                return decay #just return D0->Kpi because D0 didn't come from a D*
            # print("MOTHERS", len(mother_indices)) # there's a lot of these...
            # print(mother_indices)
            for mother_index in mother_indices:
                mother = event[mother_index]
                absPdg_mother = mother.idAbs()

                if (absPdg_mother == 413): # if mother is D*+/-
                    # if (len(mother_indices != 1)):
                    #     print("There were", len(mother_indices), "mothers in this event!")
                    # look at daughters of mother
                    if(len(mother.daughterList()) == 2):
                        d1_index = mother.daughterList()[0] #don't use daughter1() and daughter(2)
                        d2_index = mother.daughterList()[1]
                        d1 = event[d1_index]
                        d2 = event[d2_index]
                        if(not d1 or not d2):
                            return decay
                        absPdg1 = d1.idAbs()
                        absPdg2 = d2.idAbs()

                        if((absPdg1 == 421 and absPdg2 == 211) or (absPdg1 == 211 and absPdg2 == 421)): # D0 pi or pi D0
                            decay = EMesonDecayChannel.kDecayDStartoKpipi
                            break #TODO: should this break be earlier? is it possible to have multiple mothers that are D*?

            # print(event)

        return decay


    # save D0 particle info to save to THnSparse
    def getD0Info(self, particle): 
        self.D0particleinfo = particle
        return


    def getParticleAsPseudojet(self, particle):
        psjet = fj.PseudoJet(particle.px(), particle.py(), particle.pz(), particle.e())

        psjet.set_user_index(particle.index()) #should be + user_index_offset but that is 0
        # _print = PythiaParticleInfo(pythia.event[particle.index()])
        # psjet.set_user_info(_print)

        return psjet

    # check if D0 is prompt - should only send D0 that does not come from D* here
    # also assuming that D0's mother is c (direct mother, not with other generations in between)
    def checkPrompt(self, D0particle, event):

        promptness = Promptness.kUnknown

        absPdgPart = D0particle.idAbs()
        motherlist_indices = D0particle.motherList()
        # if (len(motherlist_indices) != 1):
        #     return  
        print("D0's mothers", motherlist_indices)
        for mother_index in motherlist_indices:
            mother = event[mother_index]
            absPdg_mother = mother.idAbs()
            print("D0 mother ID", absPdg_mother)

            if (absPdg_mother == 4): #charm
                # check if mother of charm is beauty
                charms_mother_indices = mother.motherList()
                print("charm's mothers", charms_mother_indices)

                # if there are no mothers???
                if len(charms_mother_indices) == 0:
                    promptness = Promptness.kPrompt
                    break

                for charms_mother_index in charms_mother_indices:
                    charms_mother = event[charms_mother_index]
                    absPdg_charms_mother = charms_mother.idAbs()
                    print("charm mother ID", absPdg_charms_mother)

                    if (absPdg_charms_mother == 4): #charm
                        promptness = Promptness.kPrompt
                        break
                    if (absPdg_charms_mother == 5): #beauty
                        promptness = Promptness.kNonPrompt
                        break
                    #else: would be unknown (if c's parentage is something but not a b...??)
                break

        return promptness

    def printD0mothers(self, particle, event, num):
        # if num == 10: #break statement
        #         print("Exited with num=10 ")
        #         return

        print("This is generation", num)

        motherlist_indices = particle.motherList()
        motherlist = [event[i].name() for i in motherlist_indices]
        motherlist_status = [event[i].status() for i in motherlist_indices]
        print("The indices are", motherlist_indices)
        print("The mothers are", motherlist)
        print("The statuss are", motherlist_status)

        if len(motherlist_indices) == 0:
            return

        for mother_index in motherlist_indices:

            # if mother_index < 5: #break statement
            #     print("Exited with mother_index of ", mother_index)
            #     break

            mother = event[mother_index]
            print("Following mother ", mother.name(), "with index", mother_index)
            self.printD0mothers(mother, event, num+1)

    # check if D0's mother is D*
    def checkD0motherIsDstar(self, D0particle, event):
        motherisDstar = False

        if (D0particle.idAbs() == 421): #D0

            mother_indices = D0particle.motherList()
            if len(mother_indices) == 1: # assuming D* is the only mother to D0
                mo1 = mother_indices[0]
                if event[mo1].idAbs() == 413: #D*
                    motherisDstar = True

        # std::cout << "is mother a Dstar?  " << motherisDstar << std::endl;
        return motherisDstar

    def getSoftPion(self, D0particle, event, jet_const_arr):
        softpion_index = -1

        Dstar_index = D0particle.motherList()[0]
        poss_softpion_indices = event[Dstar_index].daughterList()
        #TODO: check if there are only two daughters??
        for daughter_index in poss_softpion_indices:
            poss_softpion_idAbs = event[daughter_index].idAbs()
            if poss_softpion_idAbs == 211:
                softpion_index = daughter_index

        # also check that the pion is in the jet constituents
        # print("softpion index", softpion_index)
        if len(jet_const_arr) > 0:
            if (self.checkIfPartInJetConst(jet_const_arr, softpion_index, 1) == False):
                softpion_index = -1
                # print("  softpion index, softpion not in jet", softpion_index)

        return softpion_index

    #---------------------------------------------------------------
    # Initiate scaling of all histograms and print final simulation info
    #---------------------------------------------------------------
    def scale_print_final_info(self, pythia):
        # Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
        scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
        print("pythia.info.sigmaGen() is", pythia.info.sigmaGen())
        print("scale_f is", scale_f)
        #print("int(pythia.info.nAccepted())", int(pythia.info.nAccepted()))

        for jetR in self.jetR_list:
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            # print(hist_list_name)
            for h in getattr(self, hist_list_name):
            #     if 'jetlevel' in h.GetTitle():
            #         continue
                h.Scale(scale_f)

        N_rejected_hadron = int(pythia.info.nAccepted() - self.hNevents.GetBinContent(1))
        print("N total final events:", int(self.hNevents.GetBinContent(1)), "with",
              self.parton_counter, "events rejected at parton selection and",
              N_rejected_hadron, "events rejected at hadronization step")
        self.hNevents.SetBinError(1, 0)
        self.hD0Nevents.SetBinError(1, 0)
        self.hD0KpiNevents.SetBinError(1, 0)
        self.hD0KpiNjets.SetBinError(1, 0)
        self.hDstarNjets.SetBinError(1, 0)

        if self.phimeson:
            self.hphiNevents.SetBinError(1, 0)
            self.hphiNjets.SetBinError(1, 0)

        if self.make_durations:
            # Print information about the durations
            print("Average time on SD+JADE tagging: %f milliseconds per jet" % (
                      sum(self.jade_durations) / len(self.jade_durations) * 1000))
            print("Average time on WTA tagging: %f milliseconds per jet" % (
                      sum(self.wta_durations) / len(self.wta_durations) * 1000))

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('--user-seed', help='PYTHIA starting seed', default=1111, type=int)
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./',
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
                        help="Filename for the (unscaled) generated particle ROOT TTree")
    parser.add_argument('--MPIon', action='store', type=int, default=1,
                        help="MPI on or off")
    parser.add_argument('--ISRon', action='store', type=int, default=1,
                        help="ISR on or off")
    parser.add_argument('-c', '--config_file', action='store', type=str, default='config/angularity.yaml',
                        help="Path of config file for observable configurations")
    parser.add_argument('--nocharmdecay', action='store', type=int, default=0, help="'1' turns charm decays off")
    parser.add_argument('--weightON', action='store', type=int, default=0, help="'1' turns weights on")
    parser.add_argument('--leadingptcut', action='store', type=float, default=0, help="leading track pt cut")
    parser.add_argument('--replaceKP', action='store', type=int, default=0, help="'1' replaces the K/pi pairs with D0")
    parser.add_argument('--onlygg2ccbar', action='store', type=int, default=0, help="'1' runs only gg->ccbar events, '0' runs all events")
    parser.add_argument('--onlyccbar', action='store', type=int, default=0, help="'1' runs only hard->ccbar events, '0' runs all events")
    parser.add_argument('--DstarON', action='store', type=int, default=0, help="'1' looks at EEC for D* only")
    parser.add_argument('--chinitscat', action='store', type=int, default=0, help="'0' runs all events, \
                        '1' runs only hard->ccbar events, '2' runs only gg->ccbar events, '3' runs only D0->Kpi events")
    parser.add_argument('--D0withDstarON', action='store', type=int, default=0, help="'1' looks at EEC for D0 and D0 from D*")
    parser.add_argument('--difNorm', action='store', type=int, default=0, help="'1' normalizes D* with (D0+D*)")
    parser.add_argument('--softpion', action='store', type=int, default=0, help="'1' removes the soft pion from D* distribution, \
                        '2' gets only pairs of soft pion w other charged particles,'3' gets only the pair of soft pion with D0, \
                        '4' gives soft pion with everything")
    parser.add_argument('--giveptRL', action='store', type=int, default=0, help="'1' changes THnSparse to calculate pT*RL (instead of RL)")
    parser.add_argument('--runphi', action='store', type=int, default=0, help="'1' looks at the phi meson (not allowed to decay)")


    args = parser.parse_args()
    print("The arguments to run are: ", args)

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        print('File \"{0}\" does not exist! Exiting!'.format(args.config_file))
        sys.exit(0)

    # Use PYTHIA seed for event generation
    if args.user_seed < 0:
        args.user_seed = 1111

    # Have at least 1 event
    if args.nev < 1:
        args.nev = 1

    print("args for charmdecay", args.nocharmdecay)

    process = PythiaQuarkGluon(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_quark_gluon(args)
