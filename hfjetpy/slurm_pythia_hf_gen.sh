#! /bin/bash

#SBATCH --job-name="PythiaGen"
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=std
#SBATCH --time=24:00:00
#SBATCH --array=1-110
#SBATCH --output=/rstorage/alice/AnalysisResults/ang/slurm-%A_%a.out
export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/r/ruide/hfjetpy
export WORKDIR=/afs/cern.ch/work/r/ruide

# Center of mass energy in GeV
ECM=5020

FNAME="nfmodquarkcount.root" 

# Number of events per pT-hat bin (for statistics)
NEV_DESIRED=500000

# Lower edges of the pT-hat bins
#PTHAT_BINS=(9 12 16 21 28 36 45 57 70 85 100)
PTHAT_BINS=(57 100)
echo "Number of pT-hat bins: ${#PTHAT_BINS[@]}"

# Currently we have 8 nodes * 20 cores active
NCORES=110
NEV_PER_JOB=$(( $NEV_DESIRED * ${#PTHAT_BINS[@]} / $NCORES ))
echo "Number of events per job: $NEV_PER_JOB"
NCORES_PER_BIN=$(( $NCORES / ${#PTHAT_BINS[@]} ))
echo "Number of cores per pT-hat bin: $NCORES_PER_BIN"

BIN=$(( ($SLURM_ARRAY_TASK_ID - 1) / $NCORES_PER_BIN + 1))
CORE_IN_BIN=$(( ($SLURM_ARRAY_TASK_ID - 1) % $NCORES_PER_BIN + 1))
PTHAT_MIN=${PTHAT_BINS[$(( $BIN - 1 ))]}
if [ $BIN -lt ${#PTHAT_BINS[@]} ]; then
    USE_PTHAT_MAX=true
	PTHAT_MAX=${PTHAT_BINS[$BIN]}
	echo "Calculating bin $BIN (pThat=[$PTHAT_MIN,$PTHAT_MAX]) with core number $CORE_IN_BIN"
else
    USE_PTHAT_MAX=false
	echo "Calculating bin $BIN (pThat_min=$PTHAT_MIN) with core number $CORE_IN_BIN"
fi
#1111
SEED=$(( ($CORE_IN_BIN - 1) * NEV_PER_JOB + 1111))
echo SEED IS $SEED
# Do the PYTHIA simulation & matching
OUTDIR="/afs/cern.ch/work/r/ruide/lib/outBptmux13tev2"
mkdir -p $OUTDIR
module use $WORKDIR/yasp/software/modules
source $WORKDIR/yasp/venvyasp/bin/activate
module load root yasp LHAPDF6 HepMC2 pythia8/8244 fastjet
source $WORKDIR/yasp/software/root/6.28.12/bin/thisroot.sh
module load heppyy
echo "python is" $(which python)
SCRIPT="/afs/cern.ch/work/r/ruide/hfjetpy/hfjetpy/pythia_quark_gluon_ezra.py"
CONFIG="/afs/cern.ch/work/r/ruide/hfjetpy/hfjetpy/config/mass_zg_thetag.yaml"

#PTHAT_MIN minimum pt transfer
if $USE_PTHAT_MAX; then
	echo "python $SCRIPT -o $OUTDIR -c $CONFIG --user-seed $SEED --py-pthatmin\
	 $PTHAT_MIN --py-ecm $ECM --nev $NEV_PER_JOB --replaceKP 1 --chinitscat 3\
	  --pythiaopts HardQCD:all=on,PhaseSpace:pTHatMax=$PTHAT_MAX"
	python $SCRIPT -o $OUTDIR -c $CONFIG --user-seed $SEED \
		--tree-output-fname $FNAME --py-pthatmin $PTHAT_MIN --py-ecm $ECM --nev $NEV_PER_JOB \
		--replaceKP 1 --chinitscat 3 --pythiaopts HardQCD:all=on,PhaseSpace:pTHatMax=$PTHAT_MAX
else
	echo "python $SCRIPT -o $OUTDIR -c $CONFIG --user-seed $SEED --py-pthatmin\
	 $PTHAT_MIN --py-ecm $ECM --nev $NEV_PER_JOB --replaceKP 1 --chinitscat 3\
	  --pythiaopts HardQCD:all=on"
	python $SCRIPT -o $OUTDIR -c $CONFIG --user-seed $SEED \
        --tree-output-fname $FNAME --py-pthatmin $PTHAT_MIN --py-ecm $ECM --nev $NEV_PER_JOB \
        --replaceKP 1 --chinitscat 3 --pythiaopts HardQCD:all=on
fi
