import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from MLP import MLP
from SLP import SLP

file_base = ''

def converttype(content):
    for c in range(len(content)):
        content[c] = float(content[c])
    return content

def drawing(traindata, testdata, wo):
    if len(wo) < 4:
        x1,y1,d1 = zip(*traindata)
        x2,y2,d2 = zip(*testdata)
        fig1 = Figure(figsize=(3,3))
        ax1 = fig1.add_subplot(111)#2*1 1pic
        ax1.title.set_text('Traindata')
        ax1.set_xlim(min(x1)-0.5,max(x1)+0.5)
        ax1.set_ylim(min(y1)-0.5,max(y1)+0.5)
        ax1.scatter(x1,y1,c=d1)
        for w in range(len(wo.T)):
            if wo.T[w][2] == 0:
                xp = wo.T[w][0]/wo.T[w][1]
                ax1.axvline(x=xp)
            else:
                xp = np.linspace(min(x1)-0.5,max(x1)+0.5,100)
                yp = (((wo.T[w][0])-(xp*wo.T[w][1])))/(wo.T[w][2])
                ax1.plot(xp,yp)
        canvas = FigureCanvasTkAgg(fig1, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=7,column=1,columnspan=5)    
        
        fig2 = Figure(figsize=(3,3))
        ax2 = fig2.add_subplot(111)#2*1 1pic
        ax2.title.set_text('Testdata')
        ax2.set_xlim(min(x2)-0.5,max(x2)+0.5)
        ax2.set_ylim(min(y2)-0.5,max(y2)+0.5)
        ax2.scatter(x2,y2,c=d2)
        for w in range(len(wo.T)):
            if wo.T[w][2] == 0:
                xp = wo.T[w][0]/wo.T[w][1]
                ax2.axvline(x=xp)
            else:
                xp = np.linspace(min(x2)-0.5,max(x2)+0.5,100)
                yp = (((wo.T[w][0])-(xp*wo.T[w][1])))/(wo.T[w][2])
                ax2.plot(xp,yp)
        canvas = FigureCanvasTkAgg(fig2, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=7,column=6,columnspan=5)
    else:
        for d in range(len(traindata)):
            e = traindata[d][-1]
            traindata[d] = traindata[d][:4]
            traindata[d][-1] = e
        for d in range(len(testdata)):
            e = testdata[d][-1]
            testdata[d] = testdata[d][:4]
            testdata[d][-1] = e
        hx,hy,hz,hd = zip(*traindata)
        fig1 = plt.figure(figsize=(3,3))
        ax1 = fig1.add_subplot(111,projection='3d')#1*1 1pic
        ax1.title.set_text('Traindata')
        ax1.set_xlim(min(hx)-0.5,max(hx)+0.5)
        ax1.set_ylim(min(hy)-0.5,max(hy)+0.5)
        ax1.scatter(hx,hy,hz,c=hd)
        for w in range(len(wo.T)):
            xp = np.linspace(min(hx)-0.5,max(hx)+0.5,100)
            yp = np.linspace(min(hx)-0.5,max(hx)+0.5,100)
            X,Y = np.meshgrid(xp,yp)
            ax1.plot_surface(X,Y,Z=(wo.T[w][0]-X*wo.T[w][1]-Y*wo.T[w][2])/wo.T[w][3])
        canvas = FigureCanvasTkAgg(fig1, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=7,column=1,columnspan=5)

        tx,ty,tz,td = zip(*testdata)
        fig2 = plt.figure(figsize=(3,3))
        ax2 = fig2.add_subplot(111,projection='3d')#1*1 1pic
        ax2.title.set_text('Testdata')
        ax2.set_xlim(min(tx)-0.5,max(tx)+0.5)
        ax2.set_ylim(min(ty)-0.5,max(ty)+0.5)
        ax2.scatter(tx,ty,tz,c=td)
        for w in range(len(wo.T)):
            xp = np.linspace(min(tx)-0.5,max(tx)+0.5,100)
            yp = np.linspace(min(tx)-0.5,max(tx)+0.5,100)
            X,Y = np.meshgrid(xp,yp)
            ax2.plot_surface(X,Y,Z=(wo.T[w][0]-X*wo.T[w][1]-Y*wo.T[w][2])/wo.T[w][3])
        canvas = FigureCanvasTkAgg(fig2, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=7,column=6,columnspan=5)
    
def training(data,lr,ep,hn,m):
    test1 = ["5CloseS1.txt","2Ccircle1.txt","2Circle1.txt","2CloseS.txt","2CloseS2.txt","2CloseS3.txt","2cring.txt","2CS.txt","2Hcircle1.txt","2ring.txt","perceptron1.txt","perceptron2.txt"]
    test2 = ["2Circle2.txt","4satellite-6.txt","5CloseS1.txt","8OX.TXT","C3D.TXT","C10D.TXT","IRIS.TXT","Number.txt","perceptron3.txt","perceptron4.txt","wine.txt","xor.txt"]
    pctron = MLP(data,lr,ep,hn,m)
    traindata = pctron.train_data
    testdata = pctron.test_data
    if file_base in test1:
        pctron = SLP(data,lr,ep)
        traindata = pctron.train_data
        testdata = pctron.test_data 
    train_ac.set(pctron.train_perceptron())
    test_ac.set(pctron.test())
    weight.set(pctron.wo)
    wo = pctron.wo
    drawing(traindata, testdata, wo)
    
def _quit():
    root.quit()
    root.destroy()

def _readfile():
    global data,file_base
    filename = filedialog.askopenfilename()
    f = open(filename,mode='r')
    file = f.read().split('\n')
    if type(file_base) == str:
        fileentry.delete(0, 'end')
    file_base=os.path.basename(filename)
    fileentry.insert(0, file_base)
    data = [converttype(fc.split(' ')) for fc in file if fc != '']
    f.close()

def _training():
    lr = float(learnrate.get())
    ep = epoch.get()
    hn = hiddennode.get()
    m = momentum.get()
    training(data,lr,ep,hn,m)

root = tk.Tk()
root.title("Test")
root.geometry("850x600")
file_base = tk.StringVar()
learnrate = tk.StringVar()
hiddennode = tk.IntVar()
epoch = tk.IntVar()
momentum = tk.DoubleVar()

weight = tk.StringVar()
train_ac = tk.DoubleVar()
test_ac = tk.DoubleVar()

f = tkFont.Font(family='Ink Free')

tk.Label(master=root, text="Dataset: ",width=10,height=1,font=f).grid(row=0,column=0,columnspan=2)
fileentry = tk.Entry(master=root, textvariable=file_base)
fileentry.grid(row=0,column=2,columnspan=2)
tk.Label(master=root, text="Weight: ",width=20,height=1,font=f).grid(row=0,column=6,columnspan=2)
tk.Label(master=root, textvariable=weight).grid(row=0,column=8,columnspan=2)

tk.Label(master=root, text="Learnrate: ",width=10,height=1,font=f).grid(row=1,column=0,columnspan=2)
rateentry = tk.Spinbox(master=root, from_=0.01, to=20, increment=0.1, textvariable=learnrate, format="%.2f")
rateentry.grid(row=1,column=2,columnspan=2)
tk.Label(master=root, text="Training Accuracy: ",width=20,height=1,font=f).grid(row=1,column=6,columnspan=2)
tk.Label(master=root, textvariable=train_ac).grid(row=1,column=8,columnspan=2)

tk.Label(master=root, text="HiddenNode: ",width=10,height=1,font=f).grid(row=2,column=0,columnspan=2)
hiddenentry = tk.Spinbox(master=root, from_=0, to=1000, increment=1, textvariable=hiddennode)
hiddenentry.grid(row=2,column=2,columnspan=2)
tk.Label(master=root, text="Testing Accuracy: ",width=20,height=1,font=f).grid(row=2,column=6,columnspan=2)
tk.Label(master=root, textvariable=test_ac).grid(row=2,column=8,columnspan=2)

tk.Label(master=root, text="Momentum: ",width=10,height=1,font=f).grid(row=3,column=0,columnspan=2)
momententry = tk.Spinbox(master=root, from_=0.00, to=100, increment=0.1, textvariable=momentum,format="%.2f")
momententry.grid(row=3,column=2,columnspan=2)

tk.Label(master=root, text="Epoch: ",width=10,height=1,font=f).grid(row=4,column=0,columnspan=2)
epochentry = tk.Spinbox(master=root, from_=1, to=1000000, increment=10, textvariable=epoch)
epochentry.grid(row=4,column=2,columnspan=2)

readfile_btn = tk.Button(master=root, text="Choose File", command=_readfile,width=10,height=1,font=f)
readfile_btn.grid(row=0,column=4,columnspan=2)

train_btn = tk.Button(master=root, text="TRAINING!", command=_training,width=10,height=1,font=f)
train_btn.grid(row=4,column=4,columnspan=2)

quit_btn = tk.Button(master=root, text="QUIT", command=_quit,width=10,height=1,font=f)
quit_btn.grid(row=4,column=10)

tk.Label(master=root).grid(row=5,column=0)
tk.Label(master=root).grid(row=6,column=0)

root.mainloop()
