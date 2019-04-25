rm -f train_logs/*
rm -f *pickle
rm -f data/models/deleteme*

rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.181:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.176:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.177:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.163:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.182:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.218:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.235:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.159:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.157:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.237:~/vaelstmpredictor/
rsync -Pvua /home/acc/github/vaelstmpredictor/* 172.16.50.187:~/github/vaelstmpredictor/