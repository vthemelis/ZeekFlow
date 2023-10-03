from os import listdir
import pandas as pd
import numpy as np
import ipaddress
import pickle
import nltk
from sklearn.model_selection import train_test_split
import os.path
import time
from os import path
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from time import mktime
from tqdm.notebook import tqdm as tqdm
import tensorflow as tf

def readDataset(dPath):
    if not path.exists('df.csv'):
        return pd.read_csv(dPath)
    else:
        return None
                           
def FeatureExtractor(df,features):#,time_windows):
    dff=df.copy()
    dff['datetime']=pd.to_datetime(dff['ts'],unit = 's')
    dff.index=dff['datetime']
    dff=dff.groupby('sa')
    for i in features:
        #for j in time_windows:
        print(i)#,j)
        tmp_mean=dff[i].rolling('10min',min_periods=1).mean().reset_index()[i]
        tmp_std=dff[i].rolling('10min',min_periods=1).std().fillna(0).reset_index()[i]
        tmp_mean.index=df.index
        tmp_std.index=df.index
        df[f'{i}_mean'] = tmp_mean
        df[f'{i}_std'] = tmp_std
    return df
def FeatureExtractorTest(df,features):#,time_windows):
    dff=df.copy()
    dff['datetime']=pd.to_datetime(dff['ts'],unit = 's')
    dff.index=dff['datetime']
    dff=dff.groupby('sa')
    for i in features:
        print(i)
        count=dff[i].rolling('10min',min_periods=1).apply(lambda x: len(np.unique(x))).reset_index()[i]
        count.index=df.index
        df[f'{i}_count'] = count
    return df
def preprocess(df):
    if not path.exists('df.csv'):
        print(np.count_nonzero(df.isnull()))
        df['history'] = df['history'].fillna(0)
        df=df.dropna(axis=1)
        df=df.drop(['stos'],axis=1)
        print(df.columns[df.nunique() <= 1])
        if 'pr' in df.columns[df.nunique() <= 1]:
            dr=list(df.columns[df.nunique() <= 1])
            dr.remove('pr')
            df.drop(dr,axis=1,inplace=True)
        else:
            df.drop(df.columns[df.nunique() <= 1],axis=1,inplace=True)
        print(df.columns)
        print(df.head())
        df.td=pd.to_numeric(df.td)
        df.sp=pd.to_numeric(df.sp)
        print(df.dtypes)

        print(df.pr.unique())
        print(df.flg.unique())
        print('label',df.label.unique())
        print(df[df['label']<=0].flg.value_counts())
        print(df[df['label']>0].flg.value_counts())
        benign_flgs=list(df[df['label']<=0].flg.unique())
        print('benign_flgs',benign_flgs)
        with open('benignflgs.pk', 'wb') as fin:
            pickle.dump(benign_flgs, fin)
        diff=list(set(df[df['label']>0].flg.unique()).difference(benign_flgs))
        print('diff',diff)
        for unknown_flg in diff:
            bestED=999
            best=""
            for i in benign_flgs:
                ED=nltk.edit_distance(unknown_flg,i)
                if ED<bestED:
                    bestED=ED
                    best=i
            df=df.replace({unknown_flg:best})
            print(unknown_flg, 'replaced with',best)

        df=df.drop(df[df.pr=='ICMP'].index)
        df=df.drop(df[df.pr=='IGMP'].index)



        if False:
            profile = ProfileReport(df, title="Pre-Processing")
            profile.to_file("pandas_pre-processing.html")
        
        scaler = StandardScaler()

        labelencoder = LabelEncoder()

        df_benign=df.loc[df['label']<=0]
        labelencoder.fit(df_benign['flg'])
        df['flg'] = labelencoder.transform(df['flg'])
        with open('flglabelencoder.pk', 'wb') as fin:
            pickle.dump(labelencoder, fin)
        labelencoder.fit(df_benign['pr'])
        df['pr'] = labelencoder.transform(df['pr'])
        with open('prlabelencoder.pk', 'wb') as fin:
            pickle.dump(labelencoder, fin)

        #map ip addresses to ints
        srcips=df['sa'].values
        dstips=df['da'].values
        ipv6map={}
        ipv6cnt=0
        for i in range(len(srcips)):
            try:
                srcips[i]=int(ipaddress.IPv4Address(srcips[i]))
            except Exception as e:
                if srcips[i] in ipv6map.keys():
                    srcips[i]=ipv6map[srcips[i]]
                else:
                    ivp6map[srcips[i]]=ipv6cnt
                    srcips[i]=ipv6cnt
                    ipv6cnt+=1
            try:
                dstips[i]=int(ipaddress.IPv4Address(dstips[i]))
            except Exception as e:
                if dstips[i] in ipv6map.keys():
                    dstips[i]=ipv6map[dstips[i]]
                else:
                    ivp6map[dstips[i]]=ipv6cnt
                    dstips[i]=ipv6cnt
                    ipv6cnt+=1
        df.loc[:,'sa']=srcips.astype(float)
        df.loc[:,'da']=dstips.astype(float)

        #fix hexademical ports to decimal
        sport=df['sp'].values
        dport=df['dp'].values
        for i in range(len(sport)):
            try:
                sport[i]=int(sport[i])
            except:
                sport[i]=int(sport[i],16)
            try:
                dport[i]=int(dport[i])
            except:
                dport[i]=int(dport[i],16)
        df.loc[:,'sp']=sport.astype(float)
        df.loc[:,'dp']=dport.astype(float)



        df['ts'] =df['ts'].apply(lambda x: mktime(datetime.strptime(x,'%Y-%m-%d %H:%M:%S').timetuple()))
        df['te'] =df['te'].apply(lambda x: mktime(datetime.strptime(x,'%Y-%m-%d %H:%M:%S').timetuple()))
        ##################MASSIVE PROCESSING AHEAD############################
        #########SLIDING WINDOW TIME SERIES DATASET AUGMENTATION##############
        ##############LOAD DATASET INSTEAD OF RUNNING THIS####################

        print('hasnan1',df.isnull().values.any())
        df=df.sort_values(by=['sa','ts']).reset_index(drop=True)
        features = [ 'sp', 'dp', 'da']
        df=FeatureExtractorTest(df, features)
        if 'pr' in df.columns:
            features = [ 'td', 'pr', 'flg','ipkt','ibyt']
        else:
            if 'flg' in df.columns:
                features = [ 'td', 'flg','ipkt','ibyt']
            else:
                features = [ 'td','ipkt','ibyt']
        df=FeatureExtractor(df, features)#, time_windows)
        print('hasnan2',df.isnull().values.any())
        print('hasnan3',df.isnull().values.any())
        labels=df['label']
        df=df.drop(['ts','te','sa','da','dp','sp'],axis=1) #
        
        ####history oh######
        one_hot_vocabulary = ['s', 'h', 'a', 'd', 'f', 'r', 'c', 'g', 't', 'w', 'i', 'q', 'S', 'H', 'A', 'D', 'F', 'R', 'C', 'G', 'T', 'W', 'I', 'Q', '-', '^']
        ohidxes=np.eye(len(one_hot_vocabulary))
        ohDict={one_hot_vocabulary[idx]:key for idx,key in enumerate(ohidxes)}
        history_arrays = []
        padding_length = 23
        for history in tqdm(df['history']):
            if type(history)==int:
                len_hist=0
                history=""
            else:
                len_hist=len(history)
            i = padding_length - len_hist
            padded_history = history + i * '-'
            padded_history_arr = [ohDict[char] for char in padded_history]
            history_arrays.append(padded_history_arr)
        history_arrays = np.array(history_arrays)
        df['history_oh']=history_arrays.tolist()
        #df['history_oh']=df['history_oh'].apply(np.array)
        histories=df['history_oh']
        df=df.drop(['history_oh','history'],axis=1)
        #df=df.drop(['history'],axis=1)
        #####################
        
        scaler.fit(df.drop(['label'],axis=1).loc[df['label']<=0])
        df=df.drop(['label'],axis=1)
        cols=df.columns
        print(cols)


        print('hasnan4',df.isnull().values.any())
        df=scaler.transform(df)
        with open('standardscaler.pk', 'wb') as fin:
            pickle.dump(scaler, fin)
        df = pd.DataFrame(df, columns=cols)
        df['label']=labels
        df['history_oh']=histories
        #pickle.dump(history_arrays,open("histories.pkl","wb"))
        #print(dfout)
        df.to_csv('df.csv',index=False)
        #df['history_oh']=history_arrays
        df=pd.read_csv('df.csv')
        df['history_oh'] = df['history_oh'].apply(lambda x: np.array(eval(x)), 0)
        pickle.dump(df['history_oh'],open("histories.pkl","wb"))
        df=df.drop(['history_oh'],axis=1)
        df.to_csv('df.csv',index=False)
        df['history_oh']=pickle.load(open("histories.pkl","rb"))
    else:
        df=pd.read_csv('df.csv')
        #print('pkl shape',pickle.load(open("histories.pkl","rb")).shape)
        #print(df.shape)
        df['history_oh']=pickle.load(open("histories.pkl","rb"))
        #print(df.shape)
        #df['history_oh'] = df['history_oh'].apply(lambda x: np.array(eval(x)), 0)

    if False:
        profile = ProfileReport(df, title="Post-Processing")
        profile.to_file("pandas_post-processing.html")
    labelsdf=df['label']
    labels=labelsdf.values
    #print(labelsdf.value_counts())
    dataset=df.drop(['label'],axis=1) #
    #print(dataset.columns)
    if False:
        for col in dataset.columns:
            print(col)
            currcol=dataset[col]
            outl=currcol.loc[np.where(labels>0)]
            normal=currcol.loc[np.where(labels<=0)]
            print('median out: ',np.median(outl),'avg out: ',np.average(outl),'median normal: ',np.median(normal),'avg normal: ',np.average(normal))
            print('arange',currcol.min(),currcol.max())
            if currcol.min()==currcol.max():
                continue
            plt.hist((outl,normal),np.arange(currcol.min(),currcol.max(),(currcol.max()-currcol.min())/10),density=True)
            plt.show()
            fig=plt.figure()
            ax=fig.add_axes([0,0,1,1])
            ax.scatter(np.where(labels<=0), normal, color='b')
            ax.scatter(np.where(labels>0), outl, color='r')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Value')
            ax.set_title('scatter plot')
            plt.show()

    print('Columns',dataset.columns)
    vals=dataset.values
    #x_train=dataset.loc[np.where(labels==0)].values
    indices = np.arange(vals.shape[0])
    np.random.shuffle(indices) 
    vals=vals[indices]
    labels=labels[indices]
    
    x_train=vals[np.where(labels<=0)]
    #x_train=x_train.astype(np.float32)
    trainsplitpct=80
    x_train,benign_test_split=x_train[:int((trainsplitpct/100)*len(x_train)),:],x_train[int((trainsplitpct/100)*len(x_train)):,:]
    print(x_train.shape)
    print(benign_test_split.shape)

    x_test=vals[np.where(labels>0)]
    #x_test=x_test.astype(np.float32)
    return x_train,benign_test_split,x_test,labels

def preprocess_classification(df,keepOnlyNaval=False,useZeek=False):
    if not path.exists('df.csv'):
        print(np.count_nonzero(df.isnull()))
        df['history'] = df['history'].fillna(0)
        df=df.dropna(axis=1)
        df=df.drop(['stos'],axis=1)
        print(df.columns[df.nunique() <= 1])
        if 'pr' in df.columns[df.nunique() <= 1]:
            dr=list(df.columns[df.nunique() <= 1])
            dr.remove('pr')
            df.drop(dr,axis=1,inplace=True)
        else:
            df.drop(df.columns[df.nunique() <= 1],axis=1,inplace=True)
        print(df.columns)
        print(df.head())
        df.td=pd.to_numeric(df.td)
        df.sp=pd.to_numeric(df.sp)
        print(df.dtypes)

        print(df.pr.unique())
        print(df.flg.unique())
        print(df[df['label']<=0].flg.value_counts())
        print(df[df['label']>0].flg.value_counts())
        benign_flgs=list(df[df['label']<=0].flg.unique())
        with open('benignflgs.pk', 'wb') as fin:
            pickle.dump(benign_flgs, fin)
        diff=list(set(df[df['label']>0].flg.unique()).difference(benign_flgs))
        print('diff',diff)
        for unknown_flg in diff:
            bestED=999
            best=""
            for i in benign_flgs:
                ED=nltk.edit_distance(unknown_flg,i)
                if ED<bestED:
                    bestED=ED
                    best=i
            df=df.replace({unknown_flg:best})
            print(unknown_flg, 'replaced with',best)

        df=df.drop(df[df.pr=='ICMP'].index)
        df=df.drop(df[df.pr=='IGMP'].index)



        if False:
            profile = ProfileReport(df, title="Pre-Processing")
            profile.to_file("pandas_pre-processing.html")
        
        scaler = StandardScaler()

        labelencoder = LabelEncoder()

        df_benign=df.loc[df['label']<=0]
        labelencoder.fit(df_benign['flg'])
        df['flg'] = labelencoder.transform(df['flg'])
        with open('flglabelencoder.pk', 'wb') as fin:
            pickle.dump(labelencoder, fin)

        labelencoder.fit(df_benign['pr'])
        df['pr'] = labelencoder.transform(df['pr'])
        with open('prlabelencoder.pk', 'wb') as fin:
            pickle.dump(labelencoder, fin)

        #map ip addresses to ints
        srcips=df['sa'].values
        dstips=df['da'].values
        ipv6map={}
        ipv6cnt=0
        for i in range(len(srcips)):
            try:
                srcips[i]=int(ipaddress.IPv4Address(srcips[i]))
            except Exception as e:
                if srcips[i] in ipv6map.keys():
                    srcips[i]=ipv6map[srcips[i]]
                else:
                    ivp6map[srcips[i]]=ipv6cnt
                    srcips[i]=ipv6cnt
                    ipv6cnt+=1
            try:
                dstips[i]=int(ipaddress.IPv4Address(dstips[i]))
            except Exception as e:
                if dstips[i] in ipv6map.keys():
                    dstips[i]=ipv6map[dstips[i]]
                else:
                    ivp6map[dstips[i]]=ipv6cnt
                    dstips[i]=ipv6cnt
                    ipv6cnt+=1
        df.loc[:,'sa']=srcips.astype(float)
        df.loc[:,'da']=dstips.astype(float)

        #fix hexademical ports to decimal
        sport=df['sp'].values
        dport=df['dp'].values
        for i in range(len(sport)):
            try:
                sport[i]=int(sport[i])
            except:
                sport[i]=int(sport[i],16)
            try:
                dport[i]=int(dport[i])
            except:
                dport[i]=int(dport[i],16)
        df.loc[:,'sp']=sport.astype(float)
        df.loc[:,'dp']=dport.astype(float)



        df['ts'] =df['ts'].apply(lambda x: mktime(datetime.strptime(x,'%Y-%m-%d %H:%M:%S').timetuple()))
        df['te'] =df['te'].apply(lambda x: mktime(datetime.strptime(x,'%Y-%m-%d %H:%M:%S').timetuple()))
        ##################MASSIVE PROCESSING AHEAD############################
        #########SLIDING WINDOW TIME SERIES DATASET AUGMENTATION##############
        ##############LOAD DATASET INSTEAD OF RUNNING THIS####################

        print('hasnan1',df.isnull().values.any())
        df=df.sort_values(by=['sa','ts']).reset_index(drop=True)
        features = [ 'sp', 'dp', 'da']
        df=FeatureExtractorTest(df, features)
        features = [ 'td', 'pr', 'flg','ipkt','ibyt']
        df=FeatureExtractor(df, features)#, time_windows)
        print('hasnan2',df.isnull().values.any())
        print('hasnan3',df.isnull().values.any())
        labels=df['label']
        df=df.drop(['ts','te','sa','da','dp','sp'],axis=1) #
        
        ####history oh######
        one_hot_vocabulary = ['s', 'h', 'a', 'd', 'f', 'r', 'c', 'g', 't', 'w', 'i', 'q', 'S', 'H', 'A', 'D', 'F', 'R', 'C', 'G', 'T', 'W', 'I', 'Q', '-', '^']
        ohidxes=np.eye(len(one_hot_vocabulary))
        ohDict={one_hot_vocabulary[idx]:key for idx,key in enumerate(ohidxes)}
        history_arrays = []
        padding_length = 23
        for history in tqdm(df['history']):
            if type(history)==int:
                len_hist=0
                history=""
            else:
                len_hist=len(history)
            i = padding_length - len_hist
            padded_history = history + i * '-'
            padded_history_arr = [ohDict[char] for char in padded_history]
            history_arrays.append(padded_history_arr)
        history_arrays = np.array(history_arrays)
        df['history_oh']=history_arrays.tolist()
        #df['history_oh']=df['history_oh'].apply(np.array)
        histories=df['history_oh']
        df=df.drop(['history_oh','history'],axis=1)
        #df=df.drop(['history'],axis=1)
        #####################
        
        scaler.fit(df.drop(['label'],axis=1).loc[df['label']<=0])
        df=df.drop(['label'],axis=1)
        cols=df.columns
        print(cols)


        print('hasnan4',df.isnull().values.any())
        df=scaler.transform(df)
        with open('standardscaler.pk', 'wb') as fin:
            pickle.dump(scaler, fin)
        df = pd.DataFrame(df, columns=cols)
        df['label']=labels
        df['history_oh']=histories
        #pickle.dump(history_arrays,open("histories.pkl","wb"))
        #print(dfout)
        df.to_csv('df.csv',index=False)
        #df['history_oh']=history_arrays
        df=pd.read_csv('df.csv')
        df['history_oh'] = df['history_oh'].apply(lambda x: np.array(eval(x)), 0)
        pickle.dump(df['history_oh'],open("histories.pkl","wb"))
        df=df.drop(['history_oh'],axis=1)
        df.to_csv('df.csv',index=False)
        if useZeek:
            if not path.exists('df_withzeekenc.csv'):
                history_raw=pickle.load(open("histories.pkl","rb"))
                history_raw=np.moveaxis(np.stack(history_raw,axis=1),1,0)
                zeekflow=tf.keras.models.load_model("NETFLOW_full_model")
                zeek_lstm_encoder = tf.keras.Model(zeekflow.get_layer('lstminput').input, zeekflow.get_layer('lstm_2').output)
                zeek_lstm_encoder.trainable=False
                lstm_latent=zeek_lstm_encoder.predict(history_raw)
                for c in range(lstm_latent.shape[1]):
                    df['history_enc'+str(c)]=lstm_latent[:,c]
                zeekflow_encoder = tf.keras.Model(zeekflow.get_layer('lstminput').input, zeekflow.get_layer('dense_3').output)
                zeekflow_encoder.trainable=False
                zeekflow_latent=zeekflow_encoder.predict(history_raw)
                for c in range(zeekflow_latent.shape[1]):
                    df['history_zeekflow_enc'+str(c)]=zeekflow_latent[:,c]
                df.to_csv('df_withzeekenc.csv',index=False)
            else:
                df=pd.read_csv('df_withzeekenc.csv')
    else:
        df=pd.read_csv('df.csv')
        if useZeek:
            if not path.exists('df_withzeekenc.csv'):
                history_raw=pickle.load(open("histories.pkl","rb"))
                history_raw=np.moveaxis(np.stack(history_raw,axis=1),1,0)
                zeekflow=tf.keras.models.load_model("NETFLOW_full_model")
                zeekflow_encoder = tf.keras.Model([zeekflow.get_layer('lstminput').input,zeekflow.get_layer('input').input],
                                                  [zeekflow.get_layer('lstm_2').output,zeekflow.get_layer('dense_3').output])
                zeekflow_encoder.trainable=False
                zeek_latent=zeekflow_encoder.predict([history_raw,df.drop(['label'],axis=1)])
                for c in range(zeek_latent[0].shape[1]):
                    df['lstm_enc'+str(c)]=zeek_latent[0][:,c]
                for c in range(zeek_latent[1].shape[1]):
                    df['zeekflow_enc'+str(c)]=zeek_latent[1][:,c]
                df.to_csv('df_withzeekenc.csv',index=False)
                '''zeekflow_encoder = tf.keras.Model(zeekflow.get_layer('lstminput').input, zeekflow.get_layer('dense_3').output)
                zeekflow_encoder.trainable=False
                zeekflow_latent=zeekflow_encoder.predict(history_raw)
                for c in range(zeekflow_latent.shape[1]):
                    df['history_zeekflow_enc'+str(c)]=zeekflow_latent[:,c]
                df.to_csv('df_withzeekenc.csv',index=False)'''
            else:
                df=pd.read_csv('df_withzeekenc.csv')

    if False:
        profile = ProfileReport(df, title="Post-Processing")
        profile.to_file("pandas_post-processing.html")
    if keepOnlyNaval:
        df=df.loc[(df['label'] == -4) | (df['label']==14)]
        df=pd.concat([pd.concat([df[df['label']==14]]*100),df])
    labelsdf=df['label']
    labels=labelsdf.values
    print('label_value_counts',labelsdf.value_counts())
    dataset=df.drop(['label'],axis=1) #
    #print(dataset.columns)
    if False:
        for col in dataset.columns:
            print(col)
            currcol=dataset[col]
            outl=currcol.loc[np.where(labels>0)]
            normal=currcol.loc[np.where(labels<=0)]
            print('median out: ',np.median(outl),'avg out: ',np.average(outl),'median normal: ',np.median(normal),'avg normal: ',np.average(normal))
            print('arange',currcol.min(),currcol.max())
            if currcol.min()==currcol.max():
                continue
            plt.hist((outl,normal),np.arange(currcol.min(),currcol.max(),(currcol.max()-currcol.min())/10),density=True)
            plt.show()
            fig=plt.figure()
            ax=fig.add_axes([0,0,1,1])
            ax.scatter(np.where(labels<=0), normal, color='b')
            ax.scatter(np.where(labels>0), outl, color='r')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Value')
            ax.set_title('scatter plot')
            plt.show()

    print('Columns',dataset.columns)
    vals=dataset.values
    #x_train=dataset.loc[np.where(labels==0)].values
    indices = np.arange(vals.shape[0])
    np.random.shuffle(indices) 
    vals=vals[indices]
    labels=labels[indices]
    labels[np.where(labels<0)]=0
    print('label_value_counts',np.unique(labels, return_counts=True))
    #x_train=vals[np.where(labels<=0)]
    #x_train=x_train.astype(np.float32)
    trainsplitpct=80
    #x_train,x_test=vals[:int((trainsplitpct/100)*len(vals)),:],vals[int((trainsplitpct/100)*len(vals)):,:]
    #y_train,y_test=labels[:int((trainsplitpct/100)*len(labels))],labels[int((trainsplitpct/100)*len(labels)):]
    X_train, X_test, y_train, y_test=train_test_split(vals, labels, test_size=0.2, stratify=labels)
    print(X_train.shape)
    print(X_test.shape)

    #x_test=vals[np.where(labels>0)]
    #x_test=x_test.astype(np.float32)
    return X_train,y_train,X_test,y_test,dataset.columns