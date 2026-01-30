#!/bin/bash
for i in `seq 0 1`;
do
        (	
	# Learn primitives
	python sm_sb3.py --env MovingTargets-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sp_dqn/$i/

	# Moving-Targets tasks
        python sm_sb3.py --env MovingTargets-Task-1-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/zeroshot/MovingTargets-Task-1-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-Task-1-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-Task-1-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Task-1-v0/$i/
        python sb3.py --env MovingTargets-Task-1-v0 --algo dqn --skill_dir data/dqn/MovingTargets-Task-1-v0/$i/ --log_dir data/logs/dqn/MovingTargets-Task-1-v0/$i/

        python sm_sb3.py --env MovingTargets-Task-4-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/zeroshot/MovingTargets-Task-4-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-Task-4-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-Task-4-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Task-4-v0/$i/
        python sb3.py --env MovingTargets-Task-4-v0 --algo dqn --skill_dir data/dqn/MovingTargets-Task-4-v0/$i/ --log_dir data/logs/dqn/MovingTargets-Task-4-v0/$i/

        python sm_sb3.py --env MovingTargets-Multi-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/zeroshot/MovingTargets-Multi-Task-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-Multi-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-Multi-Task-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Multi-Task-v0/$i/
        python sb3.py --env MovingTargets-Multi-Task-v0 --algo dqn --skill_dir data/dqn/MovingTargets-Multi-Task-v0/$i/ --log_dir data/logs/dqn/MovingTargets-Multi-Task-v0/$i/

        python sm_sb3.py --env MovingTargets-PartiallyOrdered-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/zeroshot/MovingTargets-PartiallyOrdered-Task-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-PartiallyOrdered-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/
        python sb3.py --env MovingTargets-PartiallyOrdered-Task-v0 --algo dqn --skill_dir data/dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/

        ) &> log_moving
done &

for i in `seq 0 1`;
do
	(
	# Learn primitives
        python sm_sb3.py --env Safety-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sp_td3/Safety-v0/$i/
		
	# Safety-gym tasks
        python sm_sb3.py --env Safety-Task-1-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/zeroshot/Safety-Task-1-v0/$i/ --load
        python sm_sb3.py --env Safety-Task-1-v0 --algo td3 --sp_dir data/sp_td3/Safety-Task-1-v0/$i/ --log_dir data/logs/sm_td3/Safety-Task-1-v0/$i/
        python sb3.py --env Safety-Task-1-v0 --algo td3 --skill_dir data/td3/Safety-Task-1-v0/$i/ --log_dir data/logs/td3/Safety-Task-1-v0/$i/

        python sm_sb3.py --env Safety-Task-4-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/zeroshot/Safety-Task-4-v0/$i/ --load
        python sm_sb3.py --env Safety-Task-4-v0 --algo td3 --sp_dir data/sp_td3/Safety-Task-4-v0/$i/ --log_dir data/logs/sm_td3/Safety-Task-4-v0/$i/
        python sb3.py --env Safety-Task-4-v0 --algo td3 --skill_dir data/td3/Safety-Task-4-v0/$i/ --log_dir data/logs/td3/Safety-Task-4-v0/$i/

        python sm_sb3.py --env Safety-Multi-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/zeroshot/Safety-Multi-Task-v0/$i/ --load
        python sm_sb3.py --env Safety-Multi-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-Multi-Task-v0/$i/ --log_dir data/logs/sm_td3/Safety-Multi-Task-v0/$i/
        python sb3.py --env Safety-Multi-Task-v0 --algo td3 --skill_dir data/td3/Safety-Multi-Task-v0/$i/ --log_dir data/logs/td3/Safety-Multi-Task-v0/$i/

        python sm_sb3.py --env Safety-PartiallyOrdered-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/zeroshot/Safety-PartiallyOrdered-Task-v0/$i/ --load
        python sm_sb3.py --env Safety-PartiallyOrdered-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/sm_td3/Safety-PartiallyOrdered-Task-v0/$i/
        python sb3.py --env Safety-PartiallyOrdered-Task-v0 --algo td3 --skill_dir data/td3/Safety-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/td3/Safety-PartiallyOrdered-Task-v0/$i/   

	) &> log_safety
done &
