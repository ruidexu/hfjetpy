#!/usr/bin/env python3

"""
  Analysis IO class for jet analysis with DaVinci dataframe.

  Author: Ezra Lesser
"""

import os   # for os.path.join

# Data analysis and plotting
import ROOT

# Base class
from hfjetpy.common import common_base

################################################################
class ProcessIO(common_base.CommonBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, data_dir='/eos/lhcb/user/i/ichahrou', tree_dir='Jets',
               jet_tree_name='DecayTree', output_dir='./output',
               year_list=["2016", "2017", "2018"], magnet_list=["MU", "MD"], flavor="b",
               is_MC=True, is_pp=True, min_cent=0., max_cent=10.,
               is_jetscape=False, holes=False, event_plane_range=None,
               is_jewel=False, **kwargs):

    super(ProcessIO, self).__init__(**kwargs)
    self.data_dir = input_file
    self.output_dir = output_dir
    self.tree_dir = tree_dir
    self.jet_tree_name = jet_tree_name
    self.year_list = year_list
    self.magnet_list = magnet_list
    self.flavor = flavor
    self.is_MC = is_MC
    self.is_pp = is_pp
    self.use_ev_id_ext = use_ev_id_ext
    self.is_jetscape = is_jetscape
    self.is_jewel = is_jewel
    self.holes = holes
    self.event_plane_range = event_plane_range
    self.skip_event_tree = skip_event_tree
    self.reset_dataframes()

  #---------------------------------------------------------------
  # Clear dataframes
  #---------------------------------------------------------------
  def reset_trees(self):
    self.file_chain = None
    self.jet_tree = None

  #---------------------------------------------------------------
  # Convert ROOT TTree to SeriesGroupBy object of fastjet particles per event.
  # Optionally, define the mass assumption used in the jet reconstruction;
  #             remove a certain random fraction of tracks;
  #             randomly assign proton and kaon mass to some tracks
  #---------------------------------------------------------------
  def load_data(self, m=0.1396, offset_indices=False,
                group_by_evid=True, random_mass=False, min_pt=0.):

    self.reset_trees()

    print("Load relevant files into TChain...")

    # Chain together the appropriate objects
    chain_path = os.path.join(self.tree_dir, self.jet_tree_name)
    self.file_chain = TChain(chain_path, "")

    for year in self.year_list:
        for magnet in self.magnet_list:
            filenames = self.get_input_filenames(year, magnet)
            for filename in filenames:
                print("adding", filename)
                file = os.path.join(
                    self.data_dir, filename, self.tree_dir, self.jet_tree_name)
                self.file_chain->Add(file)

    print('Convert ROOT tree to python object...')
    print('    jet_tree_name = {}'.format(self.jet_tree_name))

    self.jet_tree = TTree(self.file_chain)

    return df_fjparticles

  #---------------------------------------------------------------
  # Generate list of DaVinci-generated input TTree filenames
  #---------------------------------------------------------------
  def get_input_filenames(self, year, magnet)

    if self.flavor != "b":
        raise NotImplementedError("Only implemented for b-jets with Jpsi2MuMu")

    filenames = []

    if self.is_MC:
        date = "04052024"
        sims = ["09k", "09l", "10a"]
        if year != 2016:
            sims += ["09h", "09i"]
        for sim in sims:
            filenames += "Bjet_MC_Jpsi2MuMu_HighPT_%s_Sim%s_%s_%s_full.root" % (year, sim, magnet, date)

    else:
        raise NotImplementedError("Only implemented for MC")

    return filenames

