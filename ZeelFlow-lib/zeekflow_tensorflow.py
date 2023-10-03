import tensorflow as tf
tf.keras.utils.set_random_seed(1)
import pandas as pd
import numpy as np
import pickle
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay,RocCurveDisplay
from sklearn.metrics import roc_auc_score
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
from keras.layers import TimeDistributed, LSTM, RepeatVector

def ZeekFlow(x_train,benign_test_split,x_test,labels,latentScale=0.5,lr=0.00003,eval_steps=3000):
    device="GPU:0"
    print(x_train.shape)
    timesteps = x_train[0][-1].shape[0]
    n_features = x_train[0][-1].shape[1]
    lstm_output_size = 10
    scaling=1
    xshape=x_train.shape[1]-1 #minus historyt
    optimizer=Adam(lr,clipnorm=1.0)

    with tf.device(device):
        lstminput = layers.Input(shape=(timesteps, n_features,), name="lstminput")
        netflow_input = layers.Input(name='input', shape=(xshape,)) 

        #lstm encoder
        x = LSTM(100, input_shape=(timesteps, n_features), return_sequences=True)(lstminput)
        x = LSTM(50, return_sequences=True)(x)
        lstm_e_output = LSTM(lstm_output_size, return_sequences=False)(x) #bottleneck

        #encoder
        x = layers.concatenate([netflow_input,lstm_e_output])
        x = layers.Dense(128*scaling)(x)
        x = layers.BatchNormalization(name='norm_1')(x)
        x = layers.LeakyReLU(name='leaky_1')(x)
        x = layers.Dense(64*scaling)(x)
        x = layers.BatchNormalization(name='norm_2')(x)
        x = layers.LeakyReLU(name='leaky_2')(x)
        x = layers.Dense(32*scaling)(x)
        x = layers.BatchNormalization(name='norm_3')(x)
        x = layers.LeakyReLU(name='leaky_3')(x)
        x = layers.Dense(int(np.floor((xshape+lstm_output_size)*latentScale)))(x)
        x = layers.BatchNormalization(name='norm_4')(x)
        x = layers.LeakyReLU(name='leaky_4')(x)

        #decoder
        y = layers.Dense(32*scaling, name='dense1')(x)
        y = layers.LeakyReLU(name='de_leaky_1')(y)
        y = layers.Dense(64*scaling, name='dense2')(y)
        y = layers.LeakyReLU(name='de_leaky_2')(y)
        y = layers.Dense(128*scaling, name='dense3')(y)
        y = layers.LeakyReLU(name='de_leaky_3')(y)
        y = layers.Dense(xshape+lstm_output_size, name='dense4')(y)
        netflow_decoded = tf.keras.layers.Lambda(lambda x: x[:,:xshape],name='netflow_decoded')(y)

        #lstm decoder
        history_z_decoded = tf.keras.layers.Lambda(lambda x: x[:,xshape:])(y)
        x = RepeatVector(timesteps)(history_z_decoded)
        y = LSTM(lstm_output_size, return_sequences=True)(x)
        y = LSTM(50, return_sequences=True)(y)
        y = LSTM(100, return_sequences=True)(y)
        y = TimeDistributed(Dense(n_features,activation='softmax'),name='history_decoded')(y)

        full_model=keras.models.Model(inputs=[lstminput,netflow_input],outputs=[y,netflow_decoded])
        full_model.compile(loss={'history_decoded':keras.losses.CategoricalCrossentropy(),'netflow_decoded':'mse'}, optimizer=optimizer,loss_weights=[10,1])
        full_model.summary()
        tf.keras.utils.plot_model(full_model,to_file='full_model.png',show_shapes=True,show_layer_names=True,show_layer_activations=True)
        ###########################TRAIN#############################
        def get_data_generator(data, batch_size=32,testing=False,returnMalLabels=False,returnBenLabels=False):
            datalen = data.shape[0]#.value
            cnt = 0
            data=data.copy()
            if returnMalLabels:
                labs=labels[np.where(labels>0)].copy()
            if returnBenLabels:
                labs=labels[np.where(labels<=0)].copy()
                #print(labs.shape,datalen,np.unique(labs, return_counts=True))
            while True:
                idxes = np.arange(datalen)
                np.random.shuffle(idxes)
                cnt += 1
                for i in range(int(np.ceil(datalen/batch_size))):
                    if returnMalLabels or returnBenLabels:
                        l=np.take(labs, idxes[i*batch_size: (i+1) * batch_size], axis=0)
                        #print('l',l)
                    train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
                    history_x=train_x[:,-1]
                    train_x=train_x[:,0:-1]
                    y = np.ones(train_x.shape[0])
                    if train_x.shape[0]<batch_size:
                        if not testing:
                            continue
                        else:
                            return
                    if returnMalLabels or returnBenLabels:
                        yield train_x.astype(np.float32),np.moveaxis(np.stack(history_x,axis=1),1,0),[y, y, y],l
                    else:
                        yield train_x.astype(np.float32),np.moveaxis(np.stack(history_x,axis=1),1,0),[y, y, y]

        print(x_train.shape)
        print(benign_test_split.shape)
        print(x_test.shape)
        niter = 500000
        batchsize = 512


        train_data_generator = get_data_generator(x_train, batchsize)

        losses=[]
        losses_zeek=[]
        losses_netflow=[]
        precision={'ZEEK':[],'NETFLOW':[],'COMBINED':[]}
        recall={'ZEEK':[],'NETFLOW':[],'COMBINED':[]}
        cutoffrange=[0.1]
        bestrecalls={'ZEEK':0,'NETFLOW':0,'COMBINED':0}
        bestprecisions={'ZEEK':0,'NETFLOW':0,'COMBINED':0}
        bestaucs={'ZEEK':0,'NETFLOW':0,'COMBINED':0}
        aucs={'ZEEK':[],'NETFLOW':[],'COMBINED':[]}
        outlierpreds_netflow_plt=[]
        normalpreds_netflow_plt=[]
        outlierpreds_zeek_plt=[]
        normalpreds_zeek_plt=[]
        outlierpreds_combined_plt=[]
        normalpreds_combined_plt=[]

        for i in tqdm(range(niter)):
            x, x_h,y = train_data_generator.__next__()      
            #print('x',x.shape)
            #print('x_h',x_h.shape)
            full_model.trainable=True
            loss = full_model.train_on_batch([x_h,x], [x_h,x])
            #print(full_model.metrics_names)
            if i % eval_steps == 0 and i!=0:
                full_model.trainable=False
                print(f'========niter: {i}========\n loss: {loss[0]}')
                losses.append(loss[0])
                losses_zeek.append(loss[1])
                losses_netflow.append(loss[2])
                malicious_test_data_generator = get_data_generator(x_test, batchsize,True,returnMalLabels=True)
                benign_test_data_generator = get_data_generator(benign_test_split, batchsize,True,returnBenLabels=True)
                outlierpreds_netflow=np.array([])
                normalpreds_netflow=np.array([])
                outlierpreds_zeek=np.array([])
                normalpreds_zeek=np.array([])

                outlierpreds_labels=np.array([])
                normalpreds_labels=np.array([])
                for genidx,gen in enumerate([malicious_test_data_generator,benign_test_data_generator]):
                    try:
                        while True:
                            if genidx==0:
                                x_t, hist_t,_,mal_label = gen.__next__()

                                p = full_model.predict([hist_t,x_t],verbose=0,steps=1)
                                outlierpreds_zeek=np.append(outlierpreds_zeek,
                                                               np.sum(np.sum(np.absolute(p[0]-hist_t),
                                                                     axis=-1),axis=-1))
                                outlierpreds_netflow=np.append(outlierpreds_netflow,
                                                               np.sum(np.absolute(p[1]-x_t),
                                                                      axis=-1))
                                outlierpreds_labels=np.append(outlierpreds_labels,mal_label)
                                #print('outlierpreds_zeek',outlierpreds_zeek.shape,outlierpreds_zeek)
                                #print('outlierpreds_netflow',outlierpreds_netflow.shape,outlierpreds_netflow)
                                #print('outlierpreds_labels',outlierpreds_labels.shape,outlierpreds_labels)

                            else:
                                x_t, hist_t,_,benign_label = gen.__next__()

                                p = full_model.predict([hist_t,x_t],verbose=0,steps=1)
                                normalpreds_zeek=np.append(normalpreds_zeek,
                                                               np.sum(np.sum(np.absolute(p[0]-hist_t),
                                                                     axis=-1),axis=-1))
                                normalpreds_netflow=np.append(normalpreds_netflow,
                                                               np.sum(np.absolute(p[1]-x_t),
                                                                      axis=-1))
                                normalpreds_labels=np.append(normalpreds_labels,benign_label)
                    except StopIteration:
                        pass
                #print('outlierpreds_zeek',outlierpreds_zeek.shape)
                #print('outlierpreds_netflow',outlierpreds_netflow.shape)
                #print('outlierpreds_labels',outlierpreds_labels.shape)
                #print(np.unique(outlierpreds_labels, return_counts=True))
                avg_zeek=(np.average(outlierpreds_zeek)+np.average(normalpreds_zeek))/2
                avg_netflow=(np.average(outlierpreds_netflow)+np.average(normalpreds_netflow))/2
                zeek_netflow_ratio=avg_zeek/avg_netflow
                outlierpreds_combined=outlierpreds_zeek+outlierpreds_netflow*zeek_netflow_ratio
                normalpreds_combined=normalpreds_zeek+normalpreds_netflow*zeek_netflow_ratio
                modality_dict={'ZEEK':(outlierpreds_zeek,normalpreds_zeek),
                                 'NETFLOW':(outlierpreds_netflow,normalpreds_netflow),
                                 'COMBINED':(outlierpreds_combined,normalpreds_combined)}
                for modality_k in modality_dict.keys():
                    outlierpreds=modality_dict[modality_k][0]
                    normalpreds=modality_dict[modality_k][1]
                    print(modality_k+' - median out: ',np.median(outlierpreds),
                      'avg out: ',np.average(outlierpreds),
                      'median normal: ',np.median(normalpreds),
                      'avg normal: ',np.average(normalpreds))
                    print(modality_k+' mean ratio:',np.average(outlierpreds)/np.average(normalpreds))
                    print(modality_k+' wasserstein_distance:',wasserstein_distance(outlierpreds,normalpreds))
                    fig=plt.figure()
                    ax=fig.add_axes([0,0,1,1])
                    s=ax.scatter(np.arange(len(normalpreds)), normalpreds, color='b',alpha=0.5,label='benign')
                    s=ax.scatter(np.arange(len(normalpreds),len(outlierpreds)+len(normalpreds)), outlierpreds, color='r',alpha=0.5,label='malicious')
                    s=ax.set_xlabel('Sample')
                    s=ax.set_ylabel('Prediction')
                    s=ax.set_title(modality_k+' scatter plot')
                    s=plt.yscale('log')
                    s=plt.legend(loc='upper left',framealpha=0.3)
                    s=plt.show()

                    fig=plt.figure()
                    ax=fig.add_axes([0,0,1,1])
                    s=ax.scatter(np.arange(len(normalpreds)),
                                 normalpreds, color='b',alpha=0.5,label='benign')
                    mUSTC=outlierpreds[np.where(outlierpreds_labels<=10)]
                    #print('mustc',len(mUSTC))
                    s=ax.scatter(np.arange(len(normalpreds),
                                           len(normalpreds)+len(mUSTC)),
                                 mUSTC, color='c',alpha=0.5,label='malUSTC')
                    mBro=outlierpreds[np.where(outlierpreds_labels==11)]
                    #print('mbro',len(mBro))
                    s=ax.scatter(np.arange(len(normalpreds)+len(mUSTC),
                                           len(normalpreds)+len(mUSTC)+len(mBro)),
                                 mBro, color='m',alpha=0.5,label='malBro')
                    mCic=outlierpreds[np.where(outlierpreds_labels>11)]
                    #print('mcic',len(mCic))
                    s=ax.scatter(np.arange(len(normalpreds)+len(mUSTC)+len(mBro),
                                           len(normalpreds)+len(mUSTC)+len(mBro)+len(mCic)),
                                 mCic, color='y',alpha=0.5,label='malCic')
                    mMyBenign=normalpreds[np.where(normalpreds_labels==-3)]
                    s=ax.scatter(np.arange(len(normalpreds)+len(mUSTC)+len(mBro)+len(mCic),
                                           len(normalpreds)+len(mUSTC)+len(mBro)+len(mCic)+len(mMyBenign)),
                                 mMyBenign, color='r',alpha=0.5,label='myB')
                    s=ax.set_xlabel('Sample')
                    s=ax.set_ylabel('Prediction')
                    s=ax.set_title(modality_k+' scatter plot (per dataset)')
                    s=plt.yscale('log')
                    s=plt.legend(loc='upper left',framealpha=0.3)
                    s=plt.show()
                    sorted_normalpreds=np.sort(normalpreds)[::-1]
                    cutoffidxes=[sorted_normalpreds[int(np.floor(len(sorted_normalpreds)*(r/100))-1)] for r in cutoffrange]
                ###
                    benign_outputs=normalpreds
                    malicious_outputs=outlierpreds
                    all_outputs=np.concatenate((benign_outputs,malicious_outputs))
                    all_gt=np.concatenate((np.zeros(len(benign_outputs)),np.ones(len(malicious_outputs))))
                    ap=average_precision_score(all_gt, all_outputs)
                    auc=roc_auc_score(all_gt, all_outputs)
                    aucs[modality_k].append(auc)
                    print(modality_k+' average precision:',ap)
                    print(modality_k+' AUC:',auc)
                    '''display = PrecisionRecallDisplay.from_predictions(all_gt, all_outputs)
                    _ = display.ax_.set_title(modality_k+" P-R curve")
                    display2 = RocCurveDisplay.from_predictions(all_gt, all_outputs)
                    _ = display2.ax_.set_title(modality_k+" Roc curve")
                    _=display2.plot()'''
                    if auc>bestaucs[modality_k]:
                        full_model.save(modality_k+'_full_model')
                        print('Saved best '+modality_k+' model at auc:',auc)
                        bestaucs[modality_k]=auc
                    for j in range(len(cutoffidxes)):
                        truepos=0
                        falsepos=0
                        truenegatives=0
                        falsenegatives=0
                        boundary=cutoffidxes[j]
                        for i in range(len(benign_outputs)):
                            if benign_outputs[i]>=boundary:
                                falsepos+=1
                            else:
                                truenegatives+=1
                        for i in range(len(malicious_outputs)):
                            if malicious_outputs[i]>=boundary:
                                truepos+=1
                            else:
                                falsenegatives+=1
                        if truepos/len(malicious_outputs)>bestrecalls[modality_k]:
                            bestrecalls[modality_k]=truepos/len(malicious_outputs)
                        if truepos/(truepos+falsepos)>bestprecisions[modality_k]:
                            bestprecisions[modality_k]=truepos/(truepos+falsepos)
                        print('% of normal wrong: ',cutoffrange[j],'/// boundary: ',round(boundary,4),'/// recall: ',round(truepos/len(malicious_outputs),4),'( best: ',round(bestrecalls[modality_k],4),') /// precision: ',round(truepos/(truepos+falsepos),4),'( best: ',round(bestprecisions[modality_k],4),') /// TP:',truepos,'FP:',falsepos,'FN:',falsenegatives,'TN:',truenegatives) # % of normal transactions wrong
                        if j==0:
                            precision[modality_k].append(truepos/(truepos+falsepos))
                            recall[modality_k].append(truepos/len(malicious_outputs))

                outlierpreds_netflow_plt.append(np.sum(outlierpreds_netflow)/outlierpreds_netflow.shape[0])
                normalpreds_netflow_plt.append(np.sum(normalpreds_netflow)/normalpreds_netflow.shape[0])
                outlierpreds_zeek_plt.append(np.sum(outlierpreds_zeek)/outlierpreds_zeek.shape[0])
                normalpreds_zeek_plt.append(np.sum(normalpreds_zeek)/normalpreds_zeek.shape[0])
                outlierpreds_combined_plt.append(outlierpreds_zeek_plt[-1]+outlierpreds_netflow_plt[-1])
                normalpreds_combined_plt.append(normalpreds_zeek_plt[-1]+normalpreds_netflow_plt[-1])

                plt.subplot(2,1,1)
                plt.title('Training loss')
                plt.plot(losses,label='combined')
                plt.plot(losses_zeek,label='zeek')
                plt.plot(losses_netflow,label='netflow')
                plt.yscale('log')
                plt.legend(loc='upper left',framealpha=0.3)
                plt.show()

                plt.subplot(2,1,1)
                plt.title('Zeek loss')
                plt.plot(normalpreds_zeek_plt,label='benign_test')
                plt.plot(outlierpreds_zeek_plt,label='malicious_test')
                plt.yscale('log')
                plt.legend(loc='upper left',framealpha=0.3)
                plt.show()

                plt.subplot(2,1,1)
                plt.title('Netflow loss')
                plt.plot(normalpreds_netflow_plt,label='benign_test')
                plt.plot(outlierpreds_netflow_plt,label='malicious_test')
                plt.yscale('log')
                plt.legend(loc='upper left',framealpha=0.3)
                plt.show()

                plt.subplot(2,1,1)
                plt.title('Combined loss')
                plt.plot(normalpreds_combined_plt,label='benign_test')
                plt.plot(outlierpreds_combined_plt,label='malicious_test')
                plt.yscale('log')
                plt.legend(loc='upper left',framealpha=0.3)
                plt.show()

                p=plt.subplot(2,1,1)
                p=plt.plot(aucs["ZEEK"],label='ZEEK AUC')
                p=plt.plot(aucs["NETFLOW"],label='NETFLOW AUC')
                p=plt.plot(aucs['COMBINED'],label='COMBINED AUC')
                p=plt.yscale('log')
                p=plt.legend(loc='upper left',framealpha=0.3)
                p=plt.show()

                plt.subplot(2,1,1)
                plt.plot(precision['ZEEK'],label='ZEEK precision')#@0.1 normal transactions wrong
                plt.plot(precision['NETFLOW'],label='NETFLOW precision')
                plt.plot(precision['COMBINED'],label='COMBINED precision')
                plt.yscale('log')
                plt.legend(loc='upper left',framealpha=0.3)
                plt.show()

                plt.subplot(2,1,1)
                plt.plot(recall['ZEEK'],label='ZEEK recall')
                plt.plot(recall['NETFLOW'],label='NETFLOW recall')
                plt.plot(recall['COMBINED'],label='COMBINED recall')
                plt.yscale('log')
                plt.legend(loc='upper left',framealpha=0.3)
                plt.show()
                
def ZeekFlowRetrain(x_train,benign_test_split,x_test,labels,latentScale=0.5,lr=0.000003,eval_steps=3000):
    full_model=tf.keras.models.load_model("NETFLOW_full_model")
    full_model.trainable=True
    ###########################TRAIN#############################
    def get_data_generator(data, batch_size=32,testing=False,returnMalLabels=False,returnBenLabels=False):
        datalen = data.shape[0]#.value
        cnt = 0
        data=data.copy()
        if returnMalLabels:
            labs=labels[np.where(labels>0)].copy()
        if returnBenLabels:
            labs=labels[np.where(labels<=0)].copy()
            #print(labs.shape,datalen,np.unique(labs, return_counts=True))
        while True:
            idxes = np.arange(datalen)
            np.random.shuffle(idxes)
            cnt += 1
            for i in range(int(np.ceil(datalen/batch_size))):
                if returnMalLabels or returnBenLabels:
                    l=np.take(labs, idxes[i*batch_size: (i+1) * batch_size], axis=0)
                    #print('l',l)
                train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
                history_x=train_x[:,-1]
                train_x=train_x[:,0:-1]
                y = np.ones(train_x.shape[0])
                if train_x.shape[0]<batch_size:
                    if not testing:
                        continue
                    else:
                        return
                if returnMalLabels or returnBenLabels:
                    yield train_x.astype(np.float32),np.moveaxis(np.stack(history_x,axis=1),1,0),[y, y, y],l
                else:
                    yield train_x.astype(np.float32),np.moveaxis(np.stack(history_x,axis=1),1,0),[y, y, y]

    print(x_train.shape)
    print(benign_test_split.shape)
    print(x_test.shape)
    niter = 500000
    batchsize = 32

    train_data_generator = get_data_generator(x_train, batchsize)

    losses=[]
    losses_zeek=[]
    losses_netflow=[]
    precision={'ZEEK':[],'NETFLOW':[],'COMBINED':[]}
    recall={'ZEEK':[],'NETFLOW':[],'COMBINED':[]}
    cutoffrange=[0.1]
    bestrecalls={'ZEEK':0,'NETFLOW':0,'COMBINED':0}
    bestprecisions={'ZEEK':0,'NETFLOW':0,'COMBINED':0}
    bestaucs={'ZEEK':0,'NETFLOW':0,'COMBINED':0}
    aucs={'ZEEK':[],'NETFLOW':[],'COMBINED':[]}

    for i in tqdm(range(niter)):
        x, x_h,y = train_data_generator.__next__()      
        #print('x',x.shape)
        #print('x_h',x_h.shape)
        full_model.trainable=True
        loss = full_model.train_on_batch([x_h,x], [x_h,x])
        #print(full_model.metrics_names)
        if i % eval_steps == 0 and i!=0:
            full_model.trainable=False
            print(f'========niter: {i}========\n loss: {loss[0]}')
            losses.append(loss[0])
            losses_zeek.append(loss[1])
            losses_netflow.append(loss[2])
            malicious_test_data_generator = get_data_generator(x_test, batchsize,True,returnMalLabels=True)
            benign_test_data_generator = get_data_generator(benign_test_split, batchsize,True,returnBenLabels=True)
            outlierpreds_netflow=np.array([])
            normalpreds_netflow=np.array([])
            outlierpreds_zeek=np.array([])
            normalpreds_zeek=np.array([])

            outlierpreds_labels=np.array([])
            normalpreds_labels=np.array([])
            for genidx,gen in enumerate([malicious_test_data_generator,benign_test_data_generator]):
                try:
                    while True:
                        if genidx==0:
                            x_t, hist_t,_,mal_label = gen.__next__()

                            p = full_model.predict([hist_t,x_t],verbose=0,steps=1)
                            outlierpreds_zeek=np.append(outlierpreds_zeek,
                                                           np.sum(np.sum(np.absolute(p[0]-hist_t),
                                                                 axis=-1),axis=-1))
                            outlierpreds_netflow=np.append(outlierpreds_netflow,
                                                           np.sum(np.absolute(p[1]-x_t),
                                                                  axis=-1))
                            outlierpreds_labels=np.append(outlierpreds_labels,mal_label)
                            #print('outlierpreds_zeek',outlierpreds_zeek.shape,outlierpreds_zeek)
                            #print('outlierpreds_netflow',outlierpreds_netflow.shape,outlierpreds_netflow)
                            #print('outlierpreds_labels',outlierpreds_labels.shape,outlierpreds_labels)

                        else:
                            x_t, hist_t,_,benign_label = gen.__next__()

                            p = full_model.predict([hist_t,x_t],verbose=0,steps=1)
                            normalpreds_zeek=np.append(normalpreds_zeek,
                                                           np.sum(np.sum(np.absolute(p[0]-hist_t),
                                                                 axis=-1),axis=-1))
                            normalpreds_netflow=np.append(normalpreds_netflow,
                                                           np.sum(np.absolute(p[1]-x_t),
                                                                  axis=-1))
                            normalpreds_labels=np.append(normalpreds_labels,benign_label)
                except StopIteration:
                    pass
            avg_zeek=(np.average(outlierpreds_zeek)+np.average(normalpreds_zeek))/2
            avg_netflow=(np.average(outlierpreds_netflow)+np.average(normalpreds_netflow))/2
            zeek_netflow_ratio=avg_zeek/avg_netflow
            outlierpreds_combined=outlierpreds_zeek+outlierpreds_netflow*zeek_netflow_ratio
            normalpreds_combined=normalpreds_zeek+normalpreds_netflow*zeek_netflow_ratio
            modality_dict={'ZEEK':(outlierpreds_zeek,normalpreds_zeek),
                             'NETFLOW':(outlierpreds_netflow,normalpreds_netflow),
                             'COMBINED':(outlierpreds_combined,normalpreds_combined)}
            for modality_k in modality_dict.keys():
                outlierpreds=modality_dict[modality_k][0]
                normalpreds=modality_dict[modality_k][1]
                print(modality_k+' - median out: ',np.median(outlierpreds),
                  'avg out: ',np.average(outlierpreds),
                  'median normal: ',np.median(normalpreds),
                  'avg normal: ',np.average(normalpreds))
                print(modality_k+' mean ratio:',np.average(outlierpreds)/np.average(normalpreds))
                print(modality_k+' wasserstein_distance:',wasserstein_distance(outlierpreds,normalpreds))

                sorted_normalpreds=np.sort(normalpreds)[::-1]
                cutoffidxes=[sorted_normalpreds[int(np.floor(len(sorted_normalpreds)*(r/100))-1)] for r in cutoffrange]
                benign_outputs=normalpreds
                malicious_outputs=outlierpreds
                all_outputs=np.concatenate((benign_outputs,malicious_outputs))
                all_gt=np.concatenate((np.zeros(len(benign_outputs)),np.ones(len(malicious_outputs))))
                ap=average_precision_score(all_gt, all_outputs)
                auc=roc_auc_score(all_gt, all_outputs)
                aucs[modality_k].append(auc)
                print(modality_k+' average precision:',ap)
                print(modality_k+' AUC:',auc)
                if auc>bestaucs[modality_k]:
                    full_model.save(modality_k+'_full_model')
                    print('Saved best '+modality_k+' model at auc:',auc)
                    bestaucs[modality_k]=auc
                for j in range(len(cutoffidxes)):
                    truepos=0
                    falsepos=0
                    truenegatives=0
                    falsenegatives=0
                    boundary=cutoffidxes[j]
                    for i in range(len(benign_outputs)):
                        if benign_outputs[i]>=boundary:
                            falsepos+=1
                        else:
                            truenegatives+=1
                    for i in range(len(malicious_outputs)):
                        if malicious_outputs[i]>=boundary:
                            truepos+=1
                        else:
                            falsenegatives+=1
                    if truepos/len(malicious_outputs)>bestrecalls[modality_k]:
                        bestrecalls[modality_k]=truepos/len(malicious_outputs)
                    if truepos/(truepos+falsepos)>bestprecisions[modality_k]:
                        bestprecisions[modality_k]=truepos/(truepos+falsepos)
                    print('% of normal wrong: ',cutoffrange[j],'/// boundary: ',round(boundary,4),'/// recall: ',round(truepos/len(malicious_outputs),4),'( best: ',round(bestrecalls[modality_k],4),') /// precision: ',round(truepos/(truepos+falsepos),4),'( best: ',round(bestprecisions[modality_k],4),') /// TP:',truepos,'FP:',falsepos,'FN:',falsenegatives,'TN:',truenegatives) # % of normal transactions wrong
                    if j==0:
                        precision[modality_k].append(truepos/(truepos+falsepos))
                        recall[modality_k].append(truepos/len(malicious_outputs))