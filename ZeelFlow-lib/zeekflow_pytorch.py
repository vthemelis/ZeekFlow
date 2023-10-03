import torch
from torch import nn
from typing import List
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class LSTM_Autoencoder(nn.Module):
    def __init__(self, netflow_input_size, timesteps, n_features, lstm_latent_size, autoencoder_latent_size, use_cuda) -> None:
        super(LSTM_Autoencoder, self).__init__()

        self.netflow_input_size = netflow_input_size
        self.timesteps = timesteps
        self.n_features = n_features
        self.lstm_latent_size = lstm_latent_size
        self.autoencoder_latent_size = autoencoder_latent_size


        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.lstm_first_layer = nn.LSTM(input_size=self.n_features, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm_second_layer = nn.LSTM(input_size=100, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm_third_layer = nn.LSTM(input_size=50, hidden_size=self.lstm_latent_size, num_layers=1, batch_first=True)
        


        self.encoder = nn.Sequential(
            nn.Linear(self.lstm_latent_size+self.netflow_input_size, 128),
            nn.BatchNorm1d(128, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(32, self.autoencoder_latent_size),
            nn.BatchNorm1d(self.autoencoder_latent_size, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.3)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.autoencoder_latent_size, 32),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(32, 64),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(128, self.lstm_latent_size+self.netflow_input_size)
        )

        self.lstm_fourth_layer = nn.LSTM(input_size=self.lstm_latent_size, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm_fifth_layer = nn.LSTM(input_size=50, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm_sixth_layer = nn.LSTM(input_size=100, hidden_size=self.n_features, num_layers=1, batch_first=True)

    def forward(self, netflow_input, zeek_input, **kwargs) -> List[torch.Tensor]:
    
        lstm_out_1, _ = self.lstm_first_layer(zeek_input)
        lstm_out_2, _ = self.lstm_second_layer(lstm_out_1)
        lstm_encoder_output, _ = self.lstm_third_layer(lstm_out_2)

        lstm_encoder_output = lstm_encoder_output[:, -1, :]
        print(lstm_encoder_output[0])

        autoencoder_input = torch.cat((netflow_input, lstm_encoder_output), 1)
        autoencoder_bottleneck = self.encoder(autoencoder_input)
        autoencoder_output = self.decoder(autoencoder_bottleneck)

        netflow_output = autoencoder_output[:, 0:20]
        zeek_output = autoencoder_output[:, 20:30]

        zeek_output = zeek_output.unsqueeze(1).repeat(1, self.timesteps, 1)
        lstm_out_4, _ = self.lstm_fourth_layer(zeek_output)
        lstm_out_5, _ = self.lstm_fifth_layer(lstm_out_4)
        zeek_reconstructed_input, _ = self.lstm_sixth_layer(lstm_out_5)

        return netflow_output, zeek_reconstructed_input

def train(model, trainloader, malicious_testloader, benign_testloader, iterations, lr):
    device = torch.device("cuda")
    test_loss = []
    precision = []
    recall=[]
    prsum=[]
    cutoffrange=np.arange(0.001,0.1,0.005)
    bestrecalls=[0 for r in range(len(cutoffrange))]
    bestprecisions=[0 for r in range(len(cutoffrange))]
    bestprecisionplusrecall=0

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_arr = []
    model.train(True)
    for i in tqdm(range(0, iterations)):
        #loss_metric = nn.MSELoss()
        loss_metric = CombinedLoss(nn.MSELoss(), nn.CrossEntropyLoss())
        model.train(True)
        optimizer.zero_grad()
        data = next(iter(trainloader))
        if torch.cuda.is_available(): 
            #data = data.to(self.device)
            netflow_data = data['netflow'].to(device)
            history_data = data['history'].to(device)
        rec_netflow_data, rec_history_data = model(netflow_data, history_data)
        rec_netflow_data = rec_netflow_data.to(device)
        rec_history_data = rec_history_data.to(device)
        loss = loss_metric(rec_netflow_data, rec_history_data, netflow_data, history_data) #loss is scalar
        loss.backward()
        optimizer.step()
        
        if i % 3000 == 0 and i!=0:
            loss_arr.append(loss.cpu().data.numpy()) # adding scalar loss to the array of losses
            #print('loss arr', loss_arr)
            malicious_outputs=[]
            benign_outputs=[]
            outlierpreds_netflow=np.array([])
            normalpreds_netflow=np.array([])
            outlierpreds_zeek=np.array([])
            normalpreds_zeek=np.array([])

            outlierpreds_labels=np.array([])
            normalpreds_labels=np.array([])
            #loss_metric = nn.MSELoss(reduction='none')
            loss_metric = CombinedLoss(nn.MSELoss(), nn.CrossEntropyLoss())
            model.eval()
            for i, data in enumerate(malicious_testloader):
                if torch.cuda.is_available():
                    #data = data.to(device)
                    #l = loss.cpu().data.numpy()
                    #a = np.average(l, axis=1)
                    netflow_data = data['netflow'].to(device)
                    history_data = data['history'].to(device)
                    rec_netflow_data, rec_history_data = model(netflow_data, history_data)
                    rec_netflow_data = rec_netflow_data.to(device)
                    rec_history_data = rec_history_data.to(device)
                    loss = loss_metric(rec_netflow_data, rec_history_data, netflow_data, history_data)

                    malicious_outputs.append(loss)
                    #outlierpreds_zeek=np.append(outlierpreds_zeek, np.sum(np.sum(np.absolute(rec_history_data.data.cpu()-history_data.data.cpu()), axis=-1),axis=-1))
                    #outlierpreds_netflow=np.append(outlierpreds_netflow, np.sum(np.absolute(rec_netflow_data.cpu()-netflow_data.cpu()), axis=-1))
                    #outlierpreds_labels=np.append(outlierpreds_labels, mal_label)

            for i, data in enumerate(benign_testloader):
                if torch.cuda.is_available():
                    #data = data.to(device)
                    netflow_data = data['netflow'].to(device)
                    history_data = data['history'].to(device)
                    rec_netflow_data, rec_history_data = model(netflow_data, history_data)
                    rec_netflow_data = rec_netflow_data.to(device)
                    rec_history_data = rec_history_data.to(device)
                    loss = loss_metric(rec_netflow_data, rec_history_data, netflow_data, history_data)
                    #loss = loss_metric(data, outputs)
                    #l = loss.cpu().data.numpy()
                    #a = np.average(l, axis=1)
                    benign_outputs.append(loss)
                    #normalpreds_zeek=np.append(normalpreds_zeek, np.sum(np.sum(np.absolute(rec_history_data-history_data), axis=-1),axis=-1))
                    #normalpreds_netflow=np.append(normalpreds_netflow,np.sum(np.absolute(rec_netflow_data-netflow_data),axis=-1))
                    #normalpreds_labels=np.append(normalpreds_labels,benign_label)
                    

            malicious_outputs=np.squeeze(malicious_outputs).flatten()
            benign_outputs=np.squeeze(benign_outputs).flatten()
            test_loss.append(np.mean(benign_outputs))
            print('median out: ',np.median(malicious_outputs),'avg out: ',np.average(malicious_outputs),'median normal: ',np.median(benign_outputs),'avg normal: ',np.average(benign_outputs))
            fig=plt.figure()
            ax=fig.add_axes([0,0,1,1])
            s=ax.scatter(np.arange(len(malicious_outputs)), malicious_outputs, color='r',alpha=0.5,label='malicious')
            s=ax.scatter(np.arange(len(benign_outputs)), benign_outputs, color='b',alpha=0.5,label='benign')
            s=ax.set_xlabel('Sample')
            s=ax.set_ylabel('Prediction')
            s=ax.set_title('scatter plot')
            #s=plt.ylim(0,100)
            s=plt.yscale('log')
            s=plt.legend()
            s=plt.show()

            sortedbenign_outputs=np.sort(benign_outputs)[::-1]
            cutoffidxes=[sortedbenign_outputs[int(np.floor(len(sortedbenign_outputs)*(r/100)))] for r in cutoffrange]

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
                if truepos/len(malicious_outputs)>bestrecalls[j]:
                    bestrecalls[j]=truepos/len(malicious_outputs)
                if truepos/(truepos+falsepos)>bestprecisions[j]:
                    bestprecisions[j]=truepos/(truepos+falsepos)
                print('% of normal wrong: ',cutoffrange[j],'/// boundary: ',round(boundary,4),'/// recall: ',round(truepos/len(malicious_outputs),4),'( best: ',round(bestrecalls[j],4),') /// precision: ',round(truepos/(truepos+falsepos),4),'( best: ',round(bestprecisions[j],4),') /// TP:',truepos,'FP:',falsepos,'FN:',falsenegatives,'TN:',truenegatives) # % of normal transactions wrong
                if j==len(cutoffidxes)-1:
                    currentprecisionrecallsum=round(truepos/len(malicious_outputs),4)+round(truepos/(truepos+falsepos),4)
                    if currentprecisionrecallsum> bestprecisionplusrecall:
                        bestprecisionplusrecall=currentprecisionrecallsum
                        torch.save(model.state_dict(), 'USTC_ae.pt')
                        print('saved best model at 0.1% of normal wrong with',round(truepos/len(malicious_outputs),4),'recall and',round(truepos/(truepos+falsepos),4),'precision')
                    precision.append(truepos/(truepos+falsepos))
                    recall.append(truepos/len(malicious_outputs))
                    prsum.append(round(truepos/len(malicious_outputs),4)+round(truepos/(truepos+falsepos),4))
            
            print('LOSS_ARR', loss_arr)
            print('TEST_LOSS:', test_loss)
            p=plt.subplot(2,1,1)
            p=plt.plot(loss_arr,label='training loss')
            p=plt.plot(test_loss,label='testing loss')
            p=plt.yscale('log')
            p=plt.legend()
            p=plt.show()

            p=plt.subplot(2,1,1)
            p=plt.plot(prsum,label='Precision+Recall @ 0.1% fp')
            p=plt.yscale('log')
            p=plt.legend()
            p=plt.show()


class ZeekFlowDataset(Dataset):
    def __init__(self, netflow_df, histories):
        self.netflow_df = netflow_df # in dataframe format
        self.histories = histories # in numpy array format

    def __getitem__(self, index):
        netflow = self.netflow_df.iloc[index].to_numpy()
        history = self.histories[index]
        #netflow = np.expand_dims(netflow, axis=0)
        #history = np.expand_dims(history, axis=0)
        return {'netflow': netflow, 'history': history}

    def __len__(self):
        return self.netflow_df.shape[0]

class CombinedLoss(nn.Module):
    def __init__(self, mse, cce):
        super().__init__()
        self.mse = mse
        self.cce = cce
        self.mse_val = 0
        self.cce_val = 0

    def forward(self, output_netflow, output_history, target_netflow, target_history):
        mse = self.mse(output_netflow, target_netflow)
        cce = self.cce(output_history, target_history)
        self.mse_val = mse 
        self.cce_val = cce
        return  mse +  10 * cce