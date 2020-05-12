import pandas as pd

alignment=open('data/ASNet/alignment.csv').readlines()
facebook_networks=pd.read_csv('data/ASNet/facebook_network.csv')
facebook_username=open('data/ASNet/facebook_username.csv').readlines()
twitter_networks=pd.read_csv('data/ASNet/twitter_network.csv')
twitter_username=open('data/ASNet/twitter_username.csv').readlines()

f_user_dict={}
t_user_dict={}
for line in facebook_username:
    user_id=line.strip("\n").split(',')[0]
    user_name=line.strip("\n").split(',')[1]
    if user_id not in f_user_dict.keys():
        f_user_dict[user_id]=[user_name]
    else:
        f_user_dict[user_id].append(user_name)
print('{} facebook users'.format(len(f_user_dict)))

for line in twitter_username:
    user_id=line.strip("\n").split(',')[0]
    user_name=line.strip("\n").split(',')[1]
    if user_id not in t_user_dict.keys():
        t_user_dict[user_id]=[user_name]
    else:
        t_user_dict[user_id].append(user_name)
print('{} twitter users'.format(len(t_user_dict)))