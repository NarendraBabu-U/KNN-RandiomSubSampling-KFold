import csv
from random import shuffle
import math
import numpy
from sklearn.cross_validation import KFold

def euclidist(l1,l2):
         sum1 = 0
         for i in range(2,len(l1)):
                  sum1= sum1+pow((float(l1[i])-float(l2[i])),2)
         return math.sqrt(sum1)

# reading the csv file and splitting it into half
f = open('wdbc.csv')
dataset = csv.reader(f)
datalist = list(dataset)
shuffle(datalist)
folds = KFold(len(datalist),5)
lables=list(sorted(set([i[1] for i in datalist])))
confmat = numpy.zeros((len(lables),len(lables)))   # confusion matrix : rows = predicted , columns = given category
accmat = []
f.close()
for trainind,testind in folds:
         trainlist = [datalist[i] for i in trainind]
         testlist = [datalist[i] for i in testind]
         dscat =[] # stores distance b/w test and train and category of train record
         pdcat =[] # stores prediction category and given test category
         
         k = 3
         for m in testlist:
                  dscat =[]
                  for j in trainlist:
                           dscat.append([euclidist(m,j),j[1]])  # [distance,traincat]
                  kminds = sorted(set([r[0]for r in dscat]))[:k] # [distance]
                  neibs = [i for i in dscat if i[0] in kminds]  # filtered fom dscat based on distance [distance,traincat]
                  cats = [i[1] for i in neibs] # [cat]
                  dists = [i[0] for i in neibs] 
                  votes = [cats.count(x) for x in cats] #[votes]
                  neibs = zip(dists,cats,votes) #[dis,cat,votes]
                  #print neibs
                  #print '-------------------------'
                  neibs = sorted(neibs, key=lambda x:(-x[2],x[0]))[:1] # sort the data first based on votes(dicresing order) after that min distance(increasing oder)
                  pdcat.append([neibs[0][1],m[1]])
         wrong = 0
         for i in pdcat:
                  confmat[lables.index(i[0])][lables.index(i[1])] += 1                  
                  if i[0]!=i[1]:
                           wrong = wrong+1
         accuracy = float((len(testlist)-wrong)) * 100 /float(len(testlist))
         accmat.append(accuracy)
print "k:",k
print "accuracy for 5-folds:"
print accmat
print "Mean",numpy.mean(accmat)
print "Varience",numpy.var(accmat)
print 'labels:',lables
print "confusion matrix : rows = predicted , columns = given category"
print confmat/5


