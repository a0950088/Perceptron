import math
import numpy as np
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self,data,learnrate,epoch,hn,moment=0.1):
        self.original_data,self.output_length,self.dlist_dict = self.normalized(data)
        self.learnrate = learnrate
        self.epoch = epoch
        self.moment = moment
        self.train_data,self.test_data = train_test_split(self.original_data, random_state=776, train_size=2/3)
        self.theta = -1
        self.outputnode = self.output_length
        self.train_input,self.train_eoutput,self.inputn = self.init_input(self.train_data)
        self.hiddennode = hn if hn > 0 else 3
        self.hiddennode_2 = 3 if self.inputn == 3 else 4
        self.train_hidden = np.ones(self.hiddennode)
        self.train_hidden_2 = np.ones(self.hiddennode_2)
        self.train_output = np.ones(self.outputnode)
        self.test_input,self.test_eoutput,_ = self.init_input(self.test_data)
        self.test_hidden = np.ones(self.hiddennode)
        self.test_hidden_2 = np.ones(self.hiddennode_2)
        self.test_output = np.ones(self.outputnode)
        self.wi = np.random.uniform(-0.2,0.2,(self.inputn,self.hiddennode))
        self.wi[0] = np.array([-1 for _ in range(self.hiddennode)])
        self.wh = np.random.uniform(-1,1,(self.hiddennode,self.hiddennode_2))
        self.wh[0] = np.array([-1 for _ in range(self.hiddennode_2)])
        self.wo = np.random.uniform(-2,2,(self.hiddennode_2,self.outputnode))
        self.wo[0] = np.array([-1 for _ in range(self.outputnode)])
        self.changei = np.zeros((self.inputn,self.hiddennode))
        self.changeh = np.zeros((self.hiddennode,self.hiddennode_2))
        self.changeo = np.zeros((self.hiddennode_2,self.outputnode))
    
    def normalized(self,data):
        original_data = data
        z = list(zip(*data))
        dlist = list(set(z[-1]))
        output_length = math.ceil(math.log(len(dlist),2))
        btlist = []
        for i in range(len(dlist)):
            bt = bin(i)[2:]
            d = [int(b) for b in bt]
            for _ in range(output_length-len(d)):
                d.insert(0,0)
            btlist.append(d)
        dlist_dict={}
        for i in range(len(dlist)):
            dlist_dict[dlist[i]] = btlist[i]
        return original_data,output_length,dlist_dict
    
    def init_input(self,input_data):
        ep = np.array([self.dlist_dict[i[-1]] for i in input_data])
        new_data = np.array([i[:-1] for i in input_data])
        new_data = np.insert(new_data,0,self.theta,axis=1)
        inputnode = len(new_data[0])
        return new_data,ep,inputnode
    
    def train_perceptron(self):
        for _ in range(self.epoch):
            En = 0.0
            for t in range(len(self.train_input)):
                self.train_hidden,self.train_hidden_2,self.train_output = self.forward(self.train_input[t],self.train_hidden,self.train_hidden_2,self.train_output)
                En += self.back_propagate(self.train_eoutput[t],self.train_input[t])
            print("En:",En)
        print("#########################################")
        suc = 0
        for t in range(len(self.train_input)):
            e = self.train_eoutput[t]
            _,_,self.train_output = self.forward(self.train_input[t],self.train_hidden,self.train_hidden_2,self.train_output)
            if self.cal_accuracy(e,self.train_output):
                suc+=1
        acc = suc/len(self.train_data)
        return acc
    
    def forward(self,input_data,hidden,hidden2,output):
        #inputlayer->hiddenlayer
        sum=0.0
        for i in range(self.hiddennode):
            sum = (input_data*self.wi.T[i]).sum()
            hidden[i] = self.sigmoid(sum)
        #muti_hiddenlayer
        sum = 0.0
        for h in range(self.hiddennode_2):
            sum = (hidden*self.wh.T[h]).sum()
            hidden2[h] = self.sigmoid(sum)
        #hiddenlayer->outputlayer
        sum=0.0
        for o in range(self.outputnode):
            sum = (hidden2*self.wo.T[o]).sum()
            output[o] = self.sigmoid(sum)
        return hidden,hidden2,output

    def sigmoid(self,x):
        return 1/(1+math.exp(-x))

    def diff_sigmoid(self,x):
        return x*(1-x)
    
    def back_propagate(self,e,input_data):
        En = 0.0
        deltao = np.zeros(self.outputnode)
        for k in range(self.outputnode):
            En = e[k]-self.train_output[k]
            deltao[k] = self.diff_sigmoid(self.train_output[k])*En
        
        deltah2 = np.zeros(self.hiddennode_2)
        for j in range(self.hiddennode_2):
            En = 0.0
            for k in range(self.outputnode):
                En += deltao[k]*self.wo[j][k]
            deltah2[j] = self.diff_sigmoid(self.train_hidden_2[j])*En
            
        deltah = np.zeros(self.hiddennode)
        for j in range(self.hiddennode):
            En = 0.0
            for k in range(self.hiddennode_2):
                En += deltah2[k]*self.wh[j][k]
            deltah[j] = self.diff_sigmoid(self.train_hidden[j])*En
        
        # change output weight
        for j in range(self.hiddennode_2):
            for k in range(self.outputnode):
                self.wo[j][k] = self.wo[j][k] + self.learnrate*(deltao[k]*self.train_hidden_2[j]) + self.moment*self.changeo[j][k]
                self.changeo[j][k] = deltao[k]*self.train_hidden_2[j]
        # change hidden weight
        for j in range(self.hiddennode):
            for k in range(self.hiddennode_2):
                self.wh[j][k] = self.wh[j][k] + self.learnrate*(deltah2[k]*self.train_hidden[j]) + self.moment*self.changeh[j][k]
                self.changeh[j][k] = deltah2[k]*self.train_hidden[j]  
        # change input weight
        for i in range(self.inputn):
            for j in range(self.hiddennode):
                self.wi[i][j] = self.wi[i][j] + self.learnrate*(deltah[j]*input_data[i]) + self.moment*self.changei[i][j]
                self.changei[i][j] = deltah[j]*input_data[i]        
        En = 0.0
        for i in range(self.outputnode):
            En += ((e[i]-self.train_output[i])**2)*0.5
        return En
    
    def test(self):
        suc = 0
        for t in range(len(self.test_input)):
            e = self.test_eoutput[t]
            _,_,self.test_output = self.forward(self.test_input[t],self.test_hidden,self.test_hidden_2,self.test_output)
            if self.cal_accuracy(e,self.test_output):
                suc+=1
        acc = suc/len(self.test_data)
        return acc
    
    
    def cal_accuracy(self,data_eout,data_trainout):
        flag = False
        for i in range(len(data_eout)):
            if data_eout[i] == 0 and data_trainout[i] <= 0.5:
                flag = True
            elif data_eout[i] == 1 and data_trainout[i] > 0.5:
                flag = True
            else:
                flag = False
                break
        if flag:
            return True
        return False
    
    