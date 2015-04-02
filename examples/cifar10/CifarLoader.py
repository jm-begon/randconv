# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:45:04 2014

@author: Jm Begon
"""

from ImageBuffer import ImageBuffer
import numpy as np
import cPickle

def unpickle(file):
    
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class CifarLightLoader(ImageBuffer):
    
    def __init__(self, databasePath,  outputFormat = ImageBuffer.PIL_FORMAT):
        self._data, self._labels = self.loadData(databasePath)
        
        self._initImageBuffer(zip(self._data, self._labels), outputFormat) 
    
    def loadData(self, filepath):
        dict = unpickle(filepath)
        data = dict["data"]
        labels = dict["labels"]
        return (data,labels)
        
    def _extract(self, imageable):
        rowData = imageable
        red = rowData[0:1024].reshape((32,32))
        green = rowData[1024:2048].reshape((32,32))
        blue = rowData[2048:3072].reshape((32,32))
        return self.mergeRGB(red, green, blue)        
        
class CifarLoader(CifarLightLoader):
    
    def __init__(self, databasePath,  outputFormat = ImageBuffer.PIL_FORMAT, labelFile=None):
        self._descr, self._filenames, self._data, self._labels = self.loadDatabase(databasePath)
        if labelFile!= None:
            self._labelNames = self.loadLabels(labelFile)
            
        self._initImageBuffer(zip(self._data, self._labels), outputFormat) 
        
    def loadDatabase(self, filepath):
        dict = unpickle(filepath)
        data = dict["data"]
        labels = dict["labels"]
        datasetDescr = dict['batch_label']
        filenames = dict['filenames']
        return (datasetDescr, filenames, data,labels)
        
    def loadLabels(self, filepath):
        meta = unpickle(filepath)
        return meta["label_names"]
        
    def getData(self, index):
        return self._data[index],self._labels[index],self._filenames[index]
        
    def get3BandsImage(self,index):
        rowData = self._data[index]
        red = rowData[0:1024].reshape((32,32))
        green = rowData[1024:2048].reshape((32,32))
        blue = rowData[2048:3072].reshape((32,32))
        return red, green, blue
        
    def mergeRGB(self, red,green,blue):
        return np.dstack((red,green,blue))
        
    
    def getImage(self,index):
        r,g,b = self.get3BandsImage(index)
        return self.mergeRGB(r,g,b)
        
    def getLabel(self, index):
        return self._labels[index]
        


class MultiCifarLoader(CifarLightLoader):
    def __init__(self, pathList, outputFormat = ImageBuffer.PIL_FORMAT):
        data = []
        labels = []
        for path in pathList:
            dataTmp, labelsTmp = self.loadData(path)
            data+=dataTmp
            labels+=labelsTmp
        
        self._data=data
        self._labels=labels
        self._initImageBuffer(zip(self._data, self._labels), outputFormat)
        
        
class CifarFromNumpies:

    def __init__(self,directory, indexFile="0index"):
        self._dir = directory
        self._index = indexFile
        
    def getFiles(self):
        with open((self._dir+self._index),"rb") as indexFile:        
            namesAndLabels = cPickle.load(indexFile)
        return [((self._dir+x),y) for x,y in namesAndLabels]

                    
if __name__ == "__main__":
    data=False
    if data:
        path = "data_batch_1"        
    else:
        path = "test_batch"
        
    labelFile = "batches.meta"
        
    cifar = CifarLoader(path, outputFormat = ImageBuffer.NUMPY_FORMAT, labelFile=labelFile)
    
    print "Size", cifar.size()
    print "Data 10", cifar.getData(10)
    
    cifar2 = cifar[0:12]
    
    print "Size", cifar2.size()
    
    for img, label in cifar2:
        print "Shape, label", img.shape, label
        pass
        
    