import numpy as np
from sklearn.model_selection import train_test_split

class SLP:
    def __init__(self,data,learnrate,epoch):
        self.data = self.normalized(data)
        self.learnrate = learnrate
        self.epoch = epoch
        self.train_data,self.test_data = train_test_split(self.data, random_state=776, train_size=2/3)
        self.theta = -1
        self.inputn = len(self.train_data[0])# input+d -> theta+input
        self.outputnode = 1
        self.train_input = np.ones(self.inputn) 
        self.train_output = np.ones(self.outputnode)
        self.test_input = np.ones(self.inputn) 
        self.test_output = np.ones(self.outputnode)
        self.wo = np.random.uniform(-1,1,(self.inputn,self.outputnode))
        self.expect_output = np.zeros(self.outputnode)
    
    def normalized(self,data):#[0-1]
        _,_,d = zip(*data)
        dmax = max(d)
        dmin = min(d)
        for i in range(len(data)):
            data[i][-1] = (data[i][-1]-dmin)/(dmax-dmin)
        return data
    
    def train_perceptron(self):
        for _ in range(self.epoch):
            for t in range(len(self.train_data)):
                self.expect_output = self.train_data[t][-1:]
                self.train_input[0] = self.theta
                self.train_input[1:] = self.train_data[t][:-1]
                tmp = 0
                for i in range(self.outputnode):
                    for j in range(self.inputn):
                        tmp += self.train_input[j]*self.wo[j][i]
                    self.train_output[i] = tmp
                self.sgn()
        return self.train_ac()
    
    def train_ac(self):
        total = len(self.train_data)
        success = 0
        for t in range(len(self.train_data)):
            self.expect_output = self.train_data[t][-1:]
            self.train_input[0] = self.theta
            self.train_input[1:] = self.train_data[t][:-1]
            tmp = 0
            for i in range(self.outputnode):
                for j in range(self.inputn):
                    tmp += self.train_input[j]*self.wo[j][i]
                self.train_output[i] = tmp
                if self.train_output[i] >= self.expect_output[i] and self.expect_output[i]==1:
                    success+=1
                elif self.train_output[i] <= self.expect_output[i] and self.expect_output[i]==0:
                    success+=1
        return success/total
    
    def sgn(self):
        for i in range(self.outputnode):
            if self.train_output[i] >= self.expect_output[i] and self.expect_output[i]==1:
                pass
            elif self.train_output[i] > self.expect_output[i] and self.expect_output[i]==0:
                self.wo = self.wo.T-(self.train_input*self.learnrate)
                self.wo = self.wo.T
            elif self.train_output[i] < self.expect_output[i] and self.expect_output[i]==1:
                self.wo = self.wo.T+(self.train_input*self.learnrate)
                self.wo = self.wo.T
            elif self.train_output[i] <= self.expect_output[i] and self.expect_output[i]==0:
                pass    
        
    def test(self):
        total = len(self.test_data)
        success = 0
        for t in range(len(self.test_data)):
            self.expect_output = self.test_data[t][-1:]
            self.test_input[0] = self.theta
            self.test_input[1:] = self.test_data[t][:-1]
            tmp = 0
            for i in range(self.outputnode):
                for j in range(self.inputn):
                    tmp += self.test_input[j]*self.wo[j][i]
                self.test_output[i] = tmp
                if self.test_output[i] >= self.expect_output[i] and self.expect_output[i]==1:
                    success+=1
                elif self.test_output[i] <= self.expect_output[i] and self.expect_output[i]==0:
                    success+=1
        return success/total