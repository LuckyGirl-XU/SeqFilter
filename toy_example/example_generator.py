import numpy as np
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
args = parser.parse_args()


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

N = 1000 
total_time_span = 1000  
timestamp_ranges = {
    '70-85': (0.7 * total_time_span, 0.85 * total_time_span),
    '85-100': (0.85 * total_time_span, 1.0 * total_time_span)
}

users = [i for i in range(10, N + 10)]
val_user_start=10+int(N*0.7)
test_user_start=10+int(N*0.85)

user_data=[]

def get_timestamps(user):
    if user>val_user_start and user<test_user_start:
        timestamp = np.random.uniform(*timestamp_ranges['70-85'])
    elif user > test_user_start:
        timestamp = np.random.uniform(*timestamp_ranges['85-100'])
    else:
        timestamp = timestamps[i]
    return timestamp

for i in range(0, len(users) - 1, 2):
    user1,user2=users[i],users[i+1]
    timestamps = []
    for _ in range(5):
        timestamp = np.random.uniform(total_time_span*0.7*i/N, total_time_span*0.7*(i+2)/N)
        timestamps.append(timestamp)
    timestamps.sort()
    items1=[0,1, 2, 3, 4]
    items2=[5,6,7,8,9]
    for j,(i1,i2) in enumerate(zip(items1,items2)):
        if j==4:
            if user1>val_user_start and user1<test_user_start:
                timestamp = np.random.uniform(*timestamp_ranges['70-85'])
                ext_roll=1
            elif user1 > test_user_start:
                timestamp = np.random.uniform(*timestamp_ranges['85-100'])
                ext_roll=2
            else:
                timestamp = timestamps[j]
                ext_roll=0
            if user1<=val_user_start:
                user_data.append((user1,i1,timestamp,ext_roll,(i1+5)))
                user_data.append((user2,i2,timestamp,ext_roll,(i2+5)%10))   
            elif user1<=test_user_start:
                if user1%4==0:
                    user_data.append((user1,i1,timestamp,ext_roll,(i1+5)))
                    last_timestamp=timestamp
                else:
                    user_data.append((user2,i2,last_timestamp,ext_roll,(i2+5)%10))   
            else:
                if user1%4==2:
                    user_data.append((user1,i1,timestamp,ext_roll,(i1+5)))
                    last_timestamp=timestamp
                else:
                    user_data.append((user2,i2,last_timestamp,ext_roll,(i2+5)%10))   
        else:
            timestamp = timestamps[j]
            ext_roll=0
            user_data.append((user1,i1,timestamp,ext_roll,(i1+5)))
            user_data.append((user2,i2,timestamp,ext_roll,(i2+5)%10))   

check_dir(f"./processed_data/{args.data}")
df = pd.DataFrame(user_data, columns=['src', 'dst', 'time', 'ext_roll','neg'])
df.to_csv(f"./processed_data/{args.data}/edges_unsorted.csv",index=False)
df = df.sort_values(by='time').reset_index(drop=True)
df.to_csv(f"./processed_data/{args.data}/edges.csv",index=False)
val_ns=np.array(df[df['ext_roll']==1]['neg']).reshape(-1,1)
test_ns=np.array(df[df['ext_roll']==2]['neg']).reshape(-1,1)
np.savez(f"./processed_data/{args.data}/val_ns.npz",ns=val_ns,num_neg_samples=1)
np.savez(f"./processed_data/{args.data}/test_ns.npz",ns=test_ns,num_neg_samples=1)