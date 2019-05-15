clear

rm -f output.file

python Reset_LAU_DeepGenerativeGenetics_SQL_Database.py --code 11235813213455

python genetic_algorithm_vae_predictor_multi_computer.py \
	--num_epochs 1 \
	--num_generations 25 \
	--population_size 10 \
	--verbose \
	--do_log \
	--do_ckpt \
	--send_back| tee output.file

rsync -Pvua output.file 172.16.50.163:/home/acc/github/vaelstmpredictor/examples/
