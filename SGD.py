import os
import re
import sys
what_type=int(sys.argv[1])
train_dataset=sys.argv[2]
test_dataset=sys.argv[3]
path = train_dataset
label1=[]
label2=[]
liarray1=[] 
liarray2=[]
tword=[]
files1=[]
files2=[]
posd=0
def bow(sl,result,di):
    #finding frequencies of each word
    li=[] 
    for i in range(len(sl)):
        li.insert(i,0)
    for w in result:
        if w in sl:
            posd=di[w]
            li[posd]+=1 
    return li
    
def bern(sl,result,di):
    #finding if the word is present in mail
    li1=[] 
    for i in range(len(sl)):
        li1.insert(i,0)
    for w in result:
        if w in sl:
            posd=di[w]
            li1[posd]=1
    return li1

for r, d, f in os.walk(path):
    for directories in d:
        for r, d, f in os.walk(path+'\\'+directories):

            for file in f:
                if '.ham' in file:
                    files1.append(os.path.join(r, file))
                    label1.append(0)
                elif '.spam' in file:
                    files1.append(os.path.join(r, file))
                    label1.append(1)
                    
for fi in files1:
    file1 = open(fi)
    content1=file1.read()
    result1 = re.split(r"\s+", content1)
    tword.extend(result1)   
s= set(tword)
sl=list(s)
di=dict()
pos=0
for w in sl:
    di[w]=pos
    pos+=1
for fi in files1:
    file1 = open(fi)
    content1=file1.read()
    
    result1 = re.split(r"\s+", content1)
    if what_type==1:
        liarray1.append(bow(sl,result1,di)) 
    elif what_type==2:
        liarray1.append(bern(sl,result1,di))  
    
    
    

#for train
path1 = test_dataset
for r, d, f in os.walk(path1):
    for directories in d:
        for r, d, f in os.walk(path1+'\\'+directories):

            for tfile in f:
                if '.ham' in tfile:
                    files2.append(os.path.join(r, tfile))
                    label2.append(0)
                elif '.spam' in tfile:
                    files2.append(os.path.join(r, tfile))
                    label2.append(1)

for fi in files2:
    file2 = open(fi)
    content2=file2.read()
    
    result2 = re.split(r"\s+", content2)
    
    liarray2.append(bow(sl,result2,di))
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


train_target = label1
train_data= liarray1


test_data = liarray2
test_target = label2

alpha = [1e-3, 1e-2, 1e-1,1e2, 1e3]
classifier = SGDClassifier()
parameters ={'alpha': alpha,'max_iter':[600,800,1000,1200],'loss':['hinge','log'],'tol':[0.001]}
clf = GridSearchCV(classifier, parameters)
clf.fit(train_data,train_target)


prediction=clf.predict(test_data)

print(np.asarray(list(prediction)))
#making a confusion matrix
cm=confusion_matrix(np.asarray(test_target),prediction)
#
print('Accuracy Score :',accuracy_score(np.asarray(test_target), prediction))
print(classification_report(np.asarray(test_target), prediction))
print(cm)
