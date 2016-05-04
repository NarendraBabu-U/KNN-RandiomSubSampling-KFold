import csv
from random import shuffle
import math
import numpy

def euclidist(l1,l2):
         sum1 = 0
         for i in range(1,len(l1)):
                  sum1= sum1+pow((float(l1[i])-float(l2[i])),2)
         return math.sqrt(sum1)
# reading the csv file and splitting it into half
f = open('wine.csv')
dataset = csv.reader(f)
datalist = list(dataset)
lables=list(sorted(set([i[0] for i in datalist])))
confmat = numpy.zeros((len(lables),len(lables)))   # confusion matrix : rows = predicted , columns = given category
accmat = []
f.close()
for iteration in range(10):
         shuffle(datalist)   # shuffle the whole data set
         trainlist = datalist[:len(datalist)/2] # copy first half into train
         testlist = datalist[len(datalist)/2:] # copy second half into test
         dscat =[] # stores distance b/w test and train and category of train record
         pdcat =[] # stores prediction category and given test category
         
         k = 3
         for m in testlist:
                  dscat =[]
                  for j in trainlist:
                           dscat.append([euclidist(m,j),j[0]])  # [distance,traincat]
                  kminds = sorted(set([r[0]for r in dscat]))[:k] # [distance]
                  neibs = [i for i in dscat if i[0] in kminds]  # filtered fom dscat based on distance [distance,traincat]
                  cats = [i[1] for i in neibs] # [cat]
                  dists = [i[0] for i in neibs] 
                  votes = [cats.count(x) for x in cats] #[votes]
                  neibs = zip(dists,cats,votes) #[dis,cat,votes]
                  #print neibs
                  #print '-------------------------'
                  neibs = sorted(neibs, key=lambda x:(-x[2],x[0]))[:1] # sort the data first based on votes(dicresing order) after that min distance(increasing oder)
                  pdcat.append([neibs[0][1],m[0]])
         wrong = 0
         for i in pdcat:
                  confmat[lables.index(i[0])][lables.index(i[1])] += 1                  
                  if i[0]!=i[1]:
                           wrong = wrong+1
         accuracy = float((len(testlist)-wrong)) * 100 /float(len(testlist))
         accmat.append(accuracy)
print "k:",k
print "accuracy of 10 iterations:"
print accmat
print "Mean",numpy.mean(accmat)
print "Varience",numpy.var(accmat)
print 'labels:',lables
print "confusion matrix : rows = predicted , columns = given category"
print confmat/10
