import os
import re
import math
import sys
import random
what_type=int(sys.argv[1])
train_dataset=sys.argv[2]
test_dataset=sys.argv[3]
path =train_dataset
#a=int(sys.argv[1])
di=dict()
files1 = []
files2 = []
files3 = []
files4 = []
files5=[]
files6=[]
files7=[]
files8=[]
files9=[]
files10=[]
setarray1=[]
setarray2=[]
setarray=[]
setarray9=[]
setarray10=[]
tword1=[]
tword2=[]
dictarray=[]
dictarray1=[]
dictarray2=[]
dictarray9=[]
dictarray10=[]

dic1=dict()
dic2=dict()
arr_cprob=[]
arrofcprob=[]
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for directories in d:
        for r, d, f in os.walk(path+'\\'+directories):

            for file in f:
                if '.ham' in file:
                    files1.append(os.path.join(r, file))
                    
                elif '.spam' in file:
                   
                    files2.append(os.path.join(r, file))
                    


def bow(result,di):
    #finding frequencies of each word
    for w in result:
        if w in di:
            di[w]=di[w]+1
        else:
            di[w]=1
    return di
    
def bern(result):
    #finding if the word is present in mail
    s1=set(result)
    S1=list(s1)
    return S1



for f in files1:
    di1=dict()
    file1 = open(f)
    content1=file1.read()
    #content of each mail
    
    result1 = re.split(r"\s+", content1)
    #words from each mail is in result
    tword1.extend(result1)
    for w in result1:
        if w in dic1:
            dic1[w]=dic1[w]+1
        else:
            dic1[w]=1
    

    dictarray1.append(bow(result1,di1))

    setarray1.append(bern(result1))

for f in files2:
    di2=dict()
    file2 = open(f)
    content2=file2.read()
    #content of each mail
    
    result2 = re.split(r"\s+", content2)
    
    #words from each mail is in result
    tword2.extend(result2)
    for w in result2:
        if w in dic2:
            dic2[w]=dic2[w]+1
        else:
            dic2[w]=1
    
    dictarray2.append(bow(result2,di2))

    setarray2.append(bern(result2))

#tword is words in all the e-mails
tword=tword1+tword2

#set contains the entire vocabulary
s= set(tword)
sl=list(s)



def tmnb(dic1,dic2,s,files1,files2):
    N=(len(files1)+len(files2))
    ham_no=len(files1)
    spam_no=len(files2)

    ham_prior=ham_no/float(N)
    spam_prior=spam_no/float(N)
    prior_proba=[ham_prior,spam_prior]
#for ham only
    T_c=0
    T_each=0
    cond_prob1=dict()
    cond_prob2=dict()
    #adding frequencies of word  in the vocabulary alon with laplace smoothing 
    for i in s:
        if i in dic1:
           
            T_c+=(dic1[i]+1)
        else:
            T_c+=1
        #for each word in vocabulary finding conditional probability
    for i in s:
        if i in dic1:
            T_each=((dic1[i]+1)/float(T_c))
            
        else:
            T_each=(1/float(T_c))
        cond_prob1[i]=T_each 
    #for spam only
    T_c1=0
    T_each1=0
    #adding frequencies of word  in the vocabulary alon with laplace smoothing
    for i in s:
        if i in dic2:
            
            T_c1+=(dic2[i]+1)
        else:
            T_c1+=1
       #for each word in vocabulary finding conditional probability
    for i in s:
        if i in dic2:
            T_each1=((dic2[i]+1)/float(T_c1))
           
        else:
            T_each1=(1/float(T_c1))
        cond_prob2[i]=T_each1 
    arr_cprob.append(cond_prob1)
    arr_cprob.append(cond_prob2)
    return s,prior_proba,arr_cprob
 
def mnb(s,prior_proba,arr_cprob,result):
    score_c=[0,0]
   
    W=[]
    #extracting words from the vocabulary that are present in mail
    for j in range(len(result)):
            if result[j] in s:
                W.append(result[j])
   
    for i in range(len(prior_proba)):
        score_c[i]=math.log(prior_proba[i])
       
        for k in range(len(W)):
            score_c[i]+=math.log(arr_cprob[i][W[k]])
             
    #returning the class of the max score  
    if score_c[0]>score_c[1]:
        return 0
    else:
        return 1
    
        
def acc(s,prior_proba,arr_cprob):
    A,B=0,0
    C,D=0,0
    path1 = test_dataset
    for r, d, f in os.walk(path1):
        for directories in d:
            for r, d, f in os.walk(path1+'\\'+directories):

                for file in f:
                    if '.ham' in file:
                        files3.append(os.path.join(r, file))
                       
                    elif '.spam' in file:
                        
                        files4.append(os.path.join(r, file))
                        
    tfile=[files3,files4]
   
    
    for i in range(0,2):
        for j in range(len(tfile[i])):
        
            file_o = open(tfile[i][j])
            content=file_o.read()
        #content of each mail
            
            result= re.split(r"\s+", content)
            label=mnb(s,prior_proba,arr_cprob,result)
            if label==0 and i==0:
                A+=1
            elif label==1 and i==0:
                B+=1
            elif label==0 and i==1:
                C+=1
            elif label==1 and i==1:
                D+=1
                
    accuracy=((A+D)/float(A+B+C+D))*100
    precision=(A/float(A+C))*100
    recall=(A/float(A+B))*100
    F_1=((2*recall*precision)/float(recall+precision))
    
    print("Accuracy of Multinomial Naive Bayes")
    print(accuracy)
    print("Precision of Multinomial Naive Bayes")
    print(precision)
    print("Recall of Multinomial Naive Bayes")
    print(recall)
    print("F1 of Multinomial Naive Bayes")
    print(F_1)
    
           
           
    

def tbnb(s,files1,files2,setarray1,setarray2):  
     

    N=(len(files1)+len(files2))
    ham_no=len(files1)
    spam_no=len(files2)
#computing prior probability
    ham_prior=ham_no/float(N)
    spam_prior=spam_no/float(N)
    Nc=[ham_no,spam_no]
    prior_proba=[ham_prior,spam_prior]
    #sarr contains unique words occuring in all the ham and spam mails
    sarr=[setarray1,setarray2]
    
    for i in range(0,2):  
        condprob1=dict()   
        for t in s:
            N_ct=0
        #counting the number of emails that consist of the word t
            for j in range(len(sarr[i])):
                    if t in sarr[i][j]:
                        N_ct+=1
            #calculating conditional probability  
            condprob1[t]=((N_ct+1)/float(Nc[i]+2))
        #appending the conditional prob for all the words in ham and then in spam
        arrofcprob.append(condprob1)
        
    return s,prior_proba,arrofcprob

    
    
    
    


def bnb(s,prior_proba,arrofcprob,data):
    Vd=[]
    scorec=[0,0]
    score=[]
    #all the words from vocabulary that are present in data as well
    for j in range(len(data)):
            if data[j] in s:
                Vd.append(data[j])
    #for each class-ham and spam
    for i in range(len(prior_proba)):
        #computing the score for each class using prior probability
        scorec[i]=math.log(prior_proba[i])
        for t in s:
            if t in Vd:
                scorec[i]+=math.log(arrofcprob[i][t]) 
                score.append(scorec[i])
            else:
                scorec[i]+=math.log(1-(arrofcprob[i][t]))
                score.append(scorec[i])
     #returning the class of the max score      
    if score[0]>score[1]:
        return 0
    else:
        return 1
def accb(s,prior_proba,arrofcprob):
    A1,B1,C1,D1=0,0,0,0
    
    path1 =test_dataset
    for r, d, f in os.walk(path1):
        for directories in d:
            for r, d, f in os.walk(path1+'\\'+directories):

                for file in f:
                    if '.ham' in file:
                        files3.append(os.path.join(r, file))
                       
                    elif '.spam' in file:
                        
                        files4.append(os.path.join(r, file))
                        
    tfile=[files3,files4]
   
    
    for i in range(0,2):
        for j in range(len(tfile[i])):
        
            file_o = open(tfile[i][j])
            content=file_o.read()
        #content of each mail
            
            data= re.split(r"\s+", content)
            label=bnb(s,prior_proba,arrofcprob,data)
            if label==0 and i==0:
                A1+=1
            elif label==1 and i==0:
                B1+=1
            elif label==0 and i==1:
                C1+=1
            elif label==1 and i==1:
                D1+=1
                
    accuracy1=((A1+D1)/float(A1+B1+C1+D1))*100
    precision1=(A1/float(A1+C1))*100
    recall1=(A1/float(A1+B1))*100
    F_2=((2*recall1*precision1)/float(recall1+precision1))
    
    print("Accuracy of Bernoulli Naive Bayes")
    print(accuracy1)
    print("Precision of Bernoulli Naive Bayes")
    print(precision1)
    print("Recall of Bernoulli Naive Bayes")
    print(recall1)
    print("F1 of Bernoulli Naive Bayes")
    print(F_2)
    



def split(files1,files2):
    L1=0
    L2=0
    L1=len(files1)*0.7
    L2=len(files2)*0.7
#70 and 30 file split
    #70 of ham
    for i in range(0,int(L1)):
        files5.append(files1[i])
    #30 of ham
    for i in range(int(L1),len(files1)):
        files7.append(files1[i])
    #70 of spam
    for i in range(0,int(L2)):    
        files6.append(files2[i])
    #30 of spam
    for i in range(int(L2),len(files2)):
        files8.append(files2[i])
    return files5,files6,files7,files8,L2,L1
        
files5,files6,files7,files8,L1,L2=split(files1,files2)
arrft=[len(files1),len(files2)]
arrf=[len(files5),len(files6)]
arrf1=[len(files7),len(files8)]
arrdict=[dictarray1,dictarray2]

arr30d=[]
#FOR 30
for i in range(0,2):
    dictA=[]
    L3=int(len(arrdict[i])*0.7)
    for j in range(L3,len(arrdict[i])):
        dictA.append(arrdict[i][j])
    arr30d.append(dictA)

        
        
#arrf=length array
def lreg(sl,arrdict,arrf,lamb1):
    n=0.01
    arrW=[]
    var1=0
    var2=0
    num=0
    x_k=0

    for i in range(len(sl)+1):
        arrW.insert(i,random.random())
#    print("arrW"+str(arrW))
    for itr in range(0,100):
        arrW1=list(arrW)
        for i in range(len(arrdict)):
            for j in range(0,arrf[i]):
                var1=0
#                print("var1"+str(var1))
                for k in range(len(sl)):
                    if sl[k] in arrdict[i][j]:
                        x_k=arrdict[i][j][sl[k]]
#                        print("arrW[k]"+str(arrW[k])+"  x_k:"+str(x_k))
                      
                        var1+=arrW[k]*x_k
#                print("var1"+str(var1))
                p_cap=0
                try:
                    num=math.exp(var1+arrW[-1])
                    p_cap=(num/float(1+num))
                except OverflowError:
#                    print("message"+str(var1+arrW[-1]))
                    p_cap=1
                
               
                var2=(i-p_cap)
                
                for k in range(len(sl)):
                    if sl[k] in arrdict[i][j]:
                        x_k=arrdict[i][j][sl[k]]
                        arrW1[k]+=(var2*x_k)
                arrW1[-1]+=var2
        for i in range(len(arrW1)):
            arrW[i]=arrW[i]+(n*arrW1[i])-(n*lamb1*arrW[i])
    return arrW

def pre(arrW,arr30d,sl,arrf1):
    Xi=0
    aA,bB,cC,dD=0,0,0,0
    for i in range(len(arr30d)):
        for j in range(len(arr30d[i])):
            x_w=0
            for k in range(len(sl)):
                if sl[k] in arr30d[i][j]:
                    Xi=arr30d[i][j][sl[k]]
                    x_w+=Xi*arrW[k]
            x_w+=arrW[-1]
            if x_w>0:
                y=1
                if i==0 and y==1:
                    bB=1
                elif i==1 and y==1:
                    dD+=1
            else:
                y=0
                if i==0 and y==0:
                    aA+=1
                elif i==1 and y==0:
                    cC+=1
    accuracy_lbow=((aA+dD)/float(aA+bB+cC+dD))*100
    precision_lbow=(aA/float(aA+cC))*100
    recall_lbow=(aA/float(aA+bB))*100
    F_1_lbow=((2*recall_lbow*precision_lbow)/float(recall_lbow+precision_lbow))
    return accuracy_lbow,precision_lbow,recall_lbow,F_1_lbow




#to create dictionaries for test
path3 =test_dataset
for r, d, f in os.walk(path3):
    for directories in d:
        for r, d, f in os.walk(path3+'\\'+directories):

            for file in f:
                if '.ham' in file:
                    files9.append(os.path.join(r, file))
                   
                elif '.spam' in file:
                    
                    files10.append(os.path.join(r, file))
for f in files9:
    di3=dict()
    file9 = open(f)
    content3=file9.read()
    #content of each mail
    result3 = re.split(r"\s+", content3)
    #words from each mail is in result
#dictionary array and set array for test data
    dictarray9.append(bow(result3,di3))
    setarray9.append(bern(result3))

for f in files10:
    di4=dict()
    file10 = open(f)
    content4=file10.read()
    #content of each mail
    result4 = re.split(r"\s+", content4) 
    #words from each mail is in result
    dictarray10.append(bow(result4,di4))
    setarray10.append(bern(result4))

arrdict1=[dictarray9,dictarray10]
arrset1=[setarray9,setarray10]
arrft1=[len(files9),len(files10)]


#arrft=[len(files1),len(files2)]
#arrf=[len(files5),len(files6)]
#arrf1=[len(files7),len(files8)]
#arrdict=[dictarray1,dictarray2]

#this is for 30 d
set30d=[]
arrset=[setarray1,setarray2]
for i in range(0,2):
    setA=[]
    L3=int(len(arrset[i])*0.7)
    for j in range(L3,len(arrset[i])):
        setA.append(arrset[i][j])
    set30d.append(setA)
    
def lregbern(sl,arrset,arrf,lamb1):
    n=0.01
    arrW=[]
    var1=0
    var2=0
    num=0
    x_k=0

    for i in range(len(sl)+1):
        arrW.insert(i,random.random())
#    print("arrW"+str(arrW))
    for itr in range(0,100):
        arrW1=list(arrW)
        for i in range(len(arrset)):
            for j in range(0,arrf[i]):
                var1=0
#                print("var1"+str(var1))
                for k in range(len(sl)):
                    if sl[k] in arrset[i][j]:
                        x_k=1
#                        print("arrW[k]"+str(arrW[k])+"  x_k:"+str(x_k))
                        var1+=arrW[k]*x_k
#                print("var1"+str(var1))
                p_cap=0
                try:
                    num=math.exp(var1+arrW[-1])
                    p_cap=(num/float(1+num))
                except OverflowError:
#                    print("message"+str(var1+arrW[-1]))
                    p_cap=1
                
               
                var2=(i-p_cap)
                
                for k in range(len(sl)):
                    if sl[k] in arrset[i][j]:
                        x_k=1
                        arrW1[k]+=(var2*x_k)
                arrW1[-1]+=var2
        for i in range(len(arrW1)):
            arrW[i]=arrW[i]+(n*arrW1[i])-(n*lamb1*arrW[i])
    return arrW

def prebern(arrW,set30d,sl,arrf1):
    Aa,Bb,Cc,Dd=0,0,0,0
    Xi=0
    for i in range(len(set30d)):
        for j in range(len(set30d[i])):
            x_w=0
            for k in range(len(sl)):
                if sl[k] in set30d[i][j]:
                    Xi=1
                    x_w+=Xi*arrW[k]
            x_w+=arrW[-1]
            if x_w>0:
                y=1
                if i==0 and y==1:
                    Bb+=1
                elif i==1 and y==1:
                    Dd+=1
            else:
                y=0
                if i==0 and y==0:
                    Aa+=1
                elif i==1 and y==0:
                    Cc+=1
    accuracy_lbern=((Aa+Dd)/float(Aa+Bb+Cc+Dd))*100
    precision_lbern=(Aa/float(Aa+Cc))*100
    recall_lbern=(Aa/float(Aa+Bb))*100
    F_1_lbern=((2*recall_lbern*precision_lbern)/float(recall_lbern+precision_lbern))
    
    return accuracy_lbern,precision_lbern,recall_lbern,F_1_lbern
if what_type==1:
    s,prior_proba,arr_cprob=tmnb(dic1,dic2,s,files1,files2)
    acc(s,prior_proba,arr_cprob) 
elif what_type==2:
    s,prior_proba,arrofcprob=tbnb(dic1,dic2,s,files1,files2)
    accb(s,prior_proba,arrofcprob)
elif what_type==3:
    #train
    lamb=[0.5,0.20,0.8,0.02,0.05]
    best_acc=[]
    for i in range(len(lamb)):
        arrW=lreg(sl,arrdict,arrft,lamb[i])
        ac,pc,rc,fc=pre(arrW,arr30d,sl,arrf1)
        best_acc.append(ac)
    ind=best_acc.index(max(best_acc))
    #test
    arrW=lreg(sl,arrdict,arrft,lamb[ind])#change 0.5 to lamb[ind]
    facc3,fpre3,frecall3,ff3=pre(arrW,arrdict1,sl,arrft1)
    print("Accuracy of LR with bow"+str(facc3))
    print("Precision of LR with bow"+str(fpre3))
    print("Recall of LR with bow"+str(frecall3))
    print("F1 of LR with bow"+str(ff3))
elif what_type==4:
    #train
    lamb=[0.2,0.50,0.8,0.02,0.05]
    best_acc1=[]
    for i in range(len(lamb)):
        arrW=lregbern(sl,arrset,arrft,lamb[i])
        fa,fp,fr,ff=prebern(arrW,set30d,sl,arrf1)
        best_acc1.append(fa)
    ind1=best_acc1.index(max(best_acc1))
    
    #test         
    arrW1=lregbern(sl,arrset,arrft,lamb[ind1])#change 0.5 to lamb[ind]
    facc1,fpre1,frecall1,ff1=prebern(arrW1,arrset1,sl,arrft1)      
    print("Accuracy of LR with bern"+str(facc1))
    print("Precision of LR with bern"+str(fpre1))
    print("Recall of LR with bern"+str(frecall1))
    print("F1 of LR with bern"+str(ff1))         




