import numpy as np
import pandas as pd
import time
from gensim.models import word2vec

class ClassBuildWord2vecTrainData:
    def __init__(self):
        """
        df: user、item交互資料
        settings: 必填USER_ID,ITEM_ID,TIMESTAMP 選填SESSION_GAP
        使用方式: ==以下內容複製貼上==
            class W2V(ClassBuildWord2vecTrainData):
                def __init__(self,df,settings={}):
                    super().__init__()
                    self.settings.update(settings)
                    self.df = df

                def writeFile(self):
                    #自定義內容
                    pass

            newW2v = W2V(dft,settings)
            newW2v.go()
        """        
        self.settings = dict()  
        self.settings['USER_ID']='USER_ID'
        self.settings['ITEM_ID']='ITEM_ID'
        self.settings['TIMESTAMP']='TIMESTAMP'
        self.settings['GROUP']='REGION_ID'
        # self.settings['AIM']='dial'
        self.settings['SESSION_GAP']=600
        self.settings['DIR_PATH']='./'
        self.settings['FILE_NAME']='w2v_train.txt'
        self.settings['MODEL_NAME']='w2v_train.model'
        
        self.settings['w2v_size'] = 32
        self.settings['w2v_epochs'] = 5
        self.settings['w2v_windows'] = 10
        self.settings['w2v_min_count'] = 5
        self.settings['w2v_max_vocab_size'] = None
        self.settings['w2v_sample'] = 1e-3
        # self.settings.update(settings) #有新的settings,採用新的
        self.trainData = []

        
    def go(self):
        """ 
        產生流程:
        self.definePath()
        self.dropDuplicates()
        self.sortDF()
        # self.markAIMtoRepeat()
        self.markNewlineWhenUserChange()
        self.markNewlineWhenTimeIntervalTooLarge()
        self.insertNewlineMark()
        self.concatString()
        self.sepByNewlineMark()
        self.writeFile()
        self.train()
        self.saveModel()
        """

        self.echoStart()
        
        self.definePath()
        self.dropDuplicates()
        self.sortDF()
        # self.markAIMtoRepeat()
        self.markNewlineWhenUserChange()
        self.markNewlineWhenTimeIntervalTooLarge()
        self.insertNewlineMark()
        self.concatString()
        self.sepByNewlineMark()
        self.buildData()
        self.train()
        self.saveModel()
        
        self.echoDone()

    def echoStart(self):
        self.time = time.time()
        print("Go...  ",end=" ")

    def echoDone(self):
        cost_time = time.time() - self.time
        print("done. (Cost %.2fs)" % cost_time,end=" ")

    def definePath(self):
        self.dir_path = self.settings['DIR_PATH']
        self.data_path = self.dir_path + '/data/'
        self.model_path = self.dir_path + '/model/'
        self.data_file = self.data_path + self.settings['FILE_NAME']
        self.model_file = self.model_path + self.settings['MODEL_NAME']

    def dropDuplicates(self):
        self.df = self.df.drop_duplicates([self.settings['USER_ID'],self.settings['ITEM_ID'],self.settings['TIMESTAMP']])

    def sortDF(self):
        self.df = self.df.sort_values([self.settings['USER_ID'],self.settings['TIMESTAMP']])

    def markAIMtoRepeat(self):
        self.df['AIM'] = self.df[self.settings['AIM']].sum(axis=1)
        self.df['repeat'] = self.df['AIM'].apply(lambda x: np.where(x>0,'#',''))
    
    def markNewlineWhenUserChange(self):
        self.df['user_new_line'] = self.df[self.settings['USER_ID']].ne(self.df[self.settings['USER_ID']].shift().bfill()).astype(int)
    
    def markNewlineWhenTimeIntervalTooLarge(self):
        self.df['gap'] = self.df[self.settings['TIMESTAMP']].diff().abs()
        self.df['time_new_line'] = self.df['gap'].apply(lambda x: np.where(x>self.settings['SESSION_GAP'],1,0))
    
    def insertNewlineMark(self):
        self.df['new_line'] = self.df[['time_new_line','user_new_line']].sum(axis=1)
        self.df['new_line_mark'] = self.df['new_line'].apply(lambda x: np.where(x>0,'@',''))
        self.df['combine'] = self.df['new_line_mark'].astype(str)+self.df[self.settings['ITEM_ID']].astype(str)
    
    def concatString(self):
        self.txt = self.df['combine'].str.cat(sep=" ")
    
    def sepByNewlineMark(self):
        self.lines = self.txt.split('@')

    def createFile(self):
        # if not os.path.isfile(self.data_file):
        open(self.data_file, 'w+', encoding='utf-8')       
    
    def lineProcess(self,line):
        return line

    def filterShortLine(self,line):
        lineToList = line.split()
        if len(lineToList)<2: 
            return True

    def otherProcess(self,line):
        return line+"\n"
    
    def writeFile(self):
        self.createFile()
        
        return True
                
    def buildData(self):
        if self.writeFile():
            # print("BuildData and writeFile...")
            with open(self.data_file, 'a+', encoding='utf-8') as f:
                for line in self.lines:

                    line = self.lineProcess(line)
                    if self.filterShortLine(line):
                        continue

                    line = self.otherProcess(line)
                    self.trainData.append(line.split())
                    f.write(line)
        else:
            # print("BuildData...")
            for line in self.lines:

                line = self.lineProcess(line)
                if self.filterShortLine(line):
                    continue

                line = self.otherProcess(line)
                self.trainData.append(line.split())
                
    def train(self):
        # print("Training Word2vec...")
        self.model = word2vec.Word2Vec(self.trainData,
                                       size=self.settings['w2v_size'],
                                       iter=self.settings['w2v_epochs'],
                                       window=self.settings['w2v_windows'],
                                       min_count=self.settings['w2v_min_count'],
                                       max_vocab_size=self.settings['w2v_max_vocab_size'],
                                       sample=self.settings['w2v_sample'],
                                       sg=1,workers=8,seed=1,alpha=0.01)
        return self.model
    
    def saveModel(self):
        self.model.save(self.model_file)
        
    """"""
    def getTrainData(self):
        with open(self.data_file) as f:
            text = f.read().splitlines()

        return [i.split() for i in text]
    
   
