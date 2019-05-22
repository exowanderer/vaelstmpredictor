clear

python Store_LAUDeepGenerativeGenetics_SQL_DB_to_CSV.py

python Reset_LAU_DeepGenerativeGenetics_SQL_Database.py --code 11235813213455

python genetic_algorithm_dense_vae_predictor_multi_computer.py \
	--run_name ga_ne50_ng25_ps100_vae_predictor_multi_computer \
	--num_epochs 50 \
	--num_generations 25 \
	--population_size 100 \
	--verbose \
	--do_log \
	--do_ckpt | ssh 172.16.50.163 "tee -a /home/acc/github/vaelstmpredictor/examples/output_ga_run_ne50_ng25_ps100_genetic_algorithm_dense_vae_predictor_multi_computer_172.16.50.187.txt"

# --send_back 
# | tee output.file
# rsync -Pvua output.file 172.16.50.163:/home/acc/github/vaelstmpredictor/examples/
