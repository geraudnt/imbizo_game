#!/bin/bash
for i in `seq 0 24`;
do
        (
        # Learn primitives
        python sm_ql.py --env Office-v0 --total_steps 100000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sp_ql/Office-v0/$i/
        python sm_ql.py --env Office-Mod-v0 --total_steps 100000 --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sp_ql/Office-Mod-v0/$i/

        # Office-World tasks
        python sm_ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-Coffee-Task-v0/$i/
        python sm_ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Coffee-Task-v0/$i/
        python sm_ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Coffee-Task-v0/$i/
        python ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Coffee-Task-v0/$i/

        python sm_ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-CoffeeMail-Task-v0/$i/
        python sm_ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-CoffeeMail-Task-v0/$i/
        python sm_ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-CoffeeMail-Task-v0/$i/
        python ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-CoffeeMail-Task-v0/$i/

        python sm_ql.py --env Office-Long-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-Long-Task-v0/$i/
        python sm_ql.py --env Office-Long-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Long-Task-v0/$i/
        python sm_ql.py --env Office-Long-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Long-Task-v0/$i/
        python ql.py --env Office-Long-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Long-Task-v0/$i/
        
	python sm_ql.py --env Office-Multi-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-Multi-Task-v0/$i/
        python sm_ql.py --env Office-Multi-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Multi-Task-v0/$i/
        python sm_ql.py --env Office-Multi-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Multi-Task-v0/$i/
        python ql.py --env Office-Multi-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Multi-Task-v0/$i/

        python sm_ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/Office-Mod-Coffee-Task-v0/$i/
        python sm_ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Mod-Coffee-Task-v0/$i/
        python sm_ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Mod-Coffee-Task-v0/$i/
        python ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Mod-Coffee-Task-v0/$i/

        python sm_ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/Office-Mod-CoffeeStrict-Task-v0/$i/
        python sm_ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Mod-CoffeeStrict-Task-v0/$i/
	python sm_ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Mod-CoffeeStrict-Task-v0/$i/
        python ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Mod-CoffeeStrict-Task-v0/$i/

        ) &> log_office &
done
