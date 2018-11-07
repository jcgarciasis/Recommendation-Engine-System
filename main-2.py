
# coding: utf-8


# # TF-recomm



import time
from collections import deque
import socket
import sys
import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import dataio
import ops




np.random.seed(13575)

BATCH_SIZE = 1000
USER_NUM = 6040
ITEM_NUM = 3952
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"




def clip(x):
    return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def get_data():
    df = dataio.read_process("./tmp/movielens/ml-1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test, rows

def get_movies():
    df = dataio.read_movies("./tmp/movielens/ml-1m/movies.dat", sep="::")
    rows = len(df)
    return df, rows


# # Data



df_train, df_test, length = get_data()
df_movies,rows = get_movies()




df_train.iloc[0:10]
df_train.describe()





print ("Movies file length: {}" .format(len(df_movies)))





df_movies.iloc[0:10]





df_movies.iloc[0].title




# # Network + train + test





samples_per_batch = len(df_train) // BATCH_SIZE

iter_train = dataio.ShuffleIterator([df_train["user"],
                                     df_train["item"],
                                    df_train["rate"]],
                                    batch_size=BATCH_SIZE)

iter_test = dataio.OneEpochIterator([df_test["user"],
                                     df_test["item"],
                                    df_test["rate"]],
                                    batch_size=-1)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM, device=DEVICE)
global_step = tf.contrib.framework.get_or_create_global_step()
_, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)






def svd(train, test,length,moviefile, trainFl=False):
    init_op = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        if trainFl == True: 
            summary_writer = tf.summary.FileWriter(logdir="./tmp/svd/log", graph=sess.graph)
            print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
            errors = deque(maxlen=samples_per_batch)
            start = time.time()
            for i in range(EPOCH_MAX * samples_per_batch):
                users, items, rates = next(iter_train)
                _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                    item_batch: items,
                                                                                                                        rate_batch: rates})
                pred_batch = clip(pred_batch)
                errors.append(np.power(pred_batch - rates, 2))
                if i % samples_per_batch == 0:
                    train_err = np.sqrt(np.mean(errors))
                    test_err2 = np.array([])
                    for users, items, rates in iter_test:
                        pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                                item_batch: items})
                        pred_batch = clip(pred_batch)
                        test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                    end = time.time()
                    test_err = np.sqrt(np.mean(test_err2))
                    print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                        end - start))
                    train_err_summary = make_scalar_summary("training_error", train_err)
                    test_err_summary = make_scalar_summary("test_error", test_err)
                    summary_writer.add_summary(train_err_summary, i)
                    summary_writer.add_summary(test_err_summary, i)
                    start = end

          

            save_path=saver.save(sess,"./tmp/")
        else: 
            print("model restored")
            saver.restore(sess, "./tmp/")




svd(df_train, df_test, length,df_movies, trainFl=False) 
print("Done!")


# # EXECUTION




def printMM(topmovies):
    print (topmovies)
    print("TOP Movies")
    for i,r in topmovies:
        print("{0:5} - {1:1.2f} - {2}" .format(i,  r, df_movies.iloc[i].title  ))





def test(train, test,length,moviefile, data, trainFl=False):
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)


        movies=list(range(len(moviefile)))
        users=[1]
        pred_batch = sess.run(infer, feed_dict={user_batch: users,item_batch: movies})
        moviesrecomm=list(zip(movies,pred_batch))
        smovies=sorted (moviesrecomm,key=lambda x:x[1],reverse=True)

        print ("\nTop Movies ------------------------------------------------------------")
        topmovies= smovies[0:10]
        printMM(topmovies)
        
        #-----------------------------------------------------------------------------
        
        print ("\n User - data {} ------------------------------------------------------------\n" .format(data))
        # give number between 1 - 5000
        del users[:]
        users.append(int(data))
        pred_batch = sess.run(infer, feed_dict={user_batch: users,item_batch: movies})
        moviesrecomm=list(zip(movies,pred_batch))
        smovies=sorted (moviesrecomm,key=lambda x:x[1],reverse=True)
        topmovies= smovies[0:10]
        printMM(topmovies)
        for item in topmovies:
            itopmovie=item[0]
            recommendedmovie=moviefile["title"][itopmovie]
            recommendedtags=moviefile["tags"][itopmovie]
    return





test(df_train, df_test, length,df_movies, data=2, trainFl=False) 


# # TO DO: 
# * Read Users 
# * Compare results for similar users 
# 
# 
# ## Users Information
# - Gender is denoted by a "M" for male and "F" for female
# - Age is chosen from the following ranges:
# 
# 	*  1:  "Under 18"
# 	* 18:  "18-24"
# 	* 25:  "25-34"
# 	* 35:  "35-44"
# 	* 45:  "45-49"
# 	* 50:  "50-55"
# 	* 56:  "56+"
# 
# - Occupation is chosen from the following choices:
# 	*  0:  "other" or not specified
# 	*  1:  "academic/educator"
# 	*  2:  "artist"
# 	*  3:  "clerical/admin"
# 	*  4:  "college/grad student"
# 	*  5:  "customer service"
# 	*  6:  "doctor/health care"
# 	*  7:  "executive/managerial"
# 	*  8:  "farmer"
# 	*  9:  "homemaker"
# 	* 10:  "K-12 student"
# 	* 11:  "lawyer"
# 	* 12:  "programmer"
# 	* 13:  "retired"
# 	* 14:  "sales/marketing"
# 	* 15:  "scientist"
# 	* 16:  "self-employed"
# 	* 17:  "technician/engineer"
# 	* 18:  "tradesman/craftsman"
# 	* 19:  "unemployed"
# 	* 20:  "writer"




import pandas as pd





def get_users():
    col_names = ["userID", "gender", "age", "occupation", "zipcode"]
    df = pd.read_csv("./tmp/movielens/ml-1m/users.dat", sep="::", header=None, names=col_names, engine='python')
    rows = len(df)
    return df, rows





df_user,rows = get_users()## Getting users from user.dat file





print(len(df_user))
df_user.describe()





df_user.iloc[0:10] #Printing users from user.dat file


# ## FIND similar users  of similar demographic





def get_users_age(age="*", gender="*", occupation="*", zipcode="*"): 
    usuar = df_user
    if age != "*":
        usuar = usuar[usuar["age"]==age]
    if gender != "*":
        usuar = usuar[usuar["gender"]==gender]
    if occupation != "*":
        usuar = usuar[usuar["occupation"]==occupation]
    if zipcode != "*":
        usuar = usuar[usuar["zipcode"]==zipcode] 

    return usuar # return the data 


# print number of users according to age gender and occupation
print(get_users_age(age=18, gender ="M", occupation = 15) )



# # Connecting main program to port 81 (Docker  TF - Recommendation engine )

def connection_port81():
    
  
    import socket
   

# Server data. Localhost and port 81
HOST = "localhost"
PORT = 81
lista = [] # A list is created to save the number of userID which are returned from TF-Recommen
# print a number of users according to age, gender and occupation. It is a list of number to ask to tensor flow container
user_IDEN = get_users_age(age=18, gender ="M", occupation = 15)
for i in range(len(user_IDEN)):
    userID = user_IDEN.iloc[i]["userID"]
    print(userID)
    lista.append(userID) #Attaching userID to a list
    


if __name__ == '__main__':
	
	# String to get data back from the server
	rec = ""



	# Counter of lineline of string(\n).
counter = 0

	# Building and connecting socket to server
cs = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
cs.connect((HOST,PORT))

	# For each user:
for user in lista:

		# Sending a encode number on utf-8
		cs.send(str(user).encode('utf-8'))

		# until the counter is not 10:
		while counter is not 10:

			# Reading a byte from response,
			byte_rec = cs.recv(1)

			# If value btye_rec is a new line, 
			# counter is incremented
			if byte_rec == b'\n':
				counter+=1

			# Otherwise, A byte is added
			# to our response.
			rec+=byte_rec.decode('utf-8')

		# printing the response
		print(rec)

		# Restarting the reset 
		# and the counter for the next user.
		rec = ""
		counter = 0

	# Closing socket
cs.close()
    
connection_port81()
    
    
