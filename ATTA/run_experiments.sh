#!/bin/bash

# Max_logit method
python main.py --cfg 'exp/atta.yaml' --dataset FS_Static  --method Max_logit
python main.py --cfg 'exp/atta.yaml' --dataset FS_LostAndFound  --method Max_logit
python main.py --cfg 'exp/atta.yaml' --dataset RoadAnomaly  --method Max_logit

# Energy method
python main.py --cfg 'exp/atta.yaml' --dataset FS_Static  --method Energy
python main.py --cfg 'exp/atta.yaml' --dataset FS_LostAndFound  --method Energy
python main.py --cfg 'exp/atta.yaml' --dataset RoadAnomaly  --method Energy

# Standardized_max_logit method
python main.py --cfg 'exp/atta.yaml' --dataset FS_Static  --method Standardized_max_logit
python main.py --cfg 'exp/atta.yaml' --dataset FS_LostAndFound  --method Standardized_max_logit
python main.py --cfg 'exp/atta.yaml' --dataset RoadAnomaly  --method Standardized_max_logit
