python ./training/q_learn.py ^
	--outputDirectory="./training/output_q_learn" ^
    --maximumHops=100 ^
    --maximumHopsPenalty=100 ^
    --numberOfUpdatesPerEpoch=1000 ^
    --validationSize=2000 ^
    --schedule="./training/schedule.csv" ^
    --nodesFilepath="./training/nodes_12_2.csv" ^
    --edgesFilepath="./training/edges_12_2.csv" ^
    --randomSeed=0