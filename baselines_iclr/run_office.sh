#!/bin/bash
#cd reward_machines
for i in `seq 0 60`; 
do
	(
	# Primitives
	python run.py --alg=smql --env=Office-new-v0 --num_timesteps=100000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --sp data/spql_$i/ --log_path=data/logs/sp-ql/Office-v0/$i
	python run.py --alg=smql --env=Office-new-up-v0 --num_timesteps=100000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --sp data/spql_up_$i/ --log_path=data/logs/sp-ql/Office-Mod-v0/$i

	# RM1 task
	python run.py --alg=smql --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Coffee-Task-v0/$i
        python run.py --alg=smql --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-Coffee-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql/Office-Coffee-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql-rs/Office-Coffee-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm/Office-Coffee-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-Coffee-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm/Office-Coffee-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm-rs/Office-Coffee-Task-v0/$i --use_rs

	# RM3 task
	python run.py --alg=smql --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-CoffeeMail-Task-v0/$i
	python run.py --alg=smql --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-CoffeeMail-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql/Office-CoffeeMail-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql-rs/Office-CoffeeMail-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm/Office-CoffeeMail-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-CoffeeMail-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm/Office-CoffeeMail-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm-rs/Office-CoffeeMail-Task-v0/$i --use_rs

	# rm4 task
	python run.py --alg=smql --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Long-Task-v0/$i
	python run.py --alg=smql --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-Long-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql/Office-Long-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql-rs/Office-Long-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm/Office-Long-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-Long-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm/Office-Long-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm-rs/Office-Long-Task-v0/$i --use_rs

	# Multi-task
	python run.py --alg=smql --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Multi-Task-v0/$i
	python run.py --alg=smql --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-Multi-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql/Office-Multi-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql-rs/Office-Multi-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm/Office-Multi-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-Multi-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm/Office-Multi-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm-rs/Office-Multi-Task-v0/$i --use_rs
		
	# Rebutal task 1
	python run.py --alg=smql --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_up_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Mod-Coffee-Task-v0/$i
	python run.py --alg=smql --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_up_$i/ --log_path=data/logs/sm-ql/fewshot/Office-Mod-Coffee-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql/Office-Mod-Coffee-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql-rs/Office-Mod-Coffee-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm/Office-Mod-Coffee-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-Mod-Coffee-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm/Office-Mod-Coffee-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm-rs/Office-Mod-Coffee-Task-v0/$i --use_rs

	# Rebutal task 2
	python run.py --alg=smql --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_up_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Mod-CoffeeStrict-Task-v0/$i
	python run.py --alg=smql --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_up_$i/ --log_path=data/logs/sm-ql/fewshot/Office-Mod-CoffeeStrict-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql/Office-Mod-CoffeeStrict-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql-rs/Office-Mod-CoffeeStrict-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm/Office-Mod-CoffeeStrict-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-Mod-CoffeeStrict-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm/Office-Mod-CoffeeStrict-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-up-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm-rs/Office-Mod-CoffeeStrict-Task-v0/$i --use_rs
		
	) &> log &
done
