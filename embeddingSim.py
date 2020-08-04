class EmbeddingSim():
    def __init__(self,df,compareDict,param={'n_sample':10}):    
        self.df = df
        self.compareDict = compareDict
        self.n_sample = param['n_sample']
        self.groupby_col = param['groupby_col']
        self.ITEM_ID = param['ITEM_ID']
        self.group_data = {sec:g for sec, g in self.df.groupby(self.groupby_col) if len(g)>0}

    def sec_cos(self,s):
        datas = {}
        key = s['key1']
        key1 = key.split("_")[1]
        i = list(self.compareDict.keys()).index(key1)
        if key1 not in self.group_data.keys():
            return datas
        col_name = self.compareDict[key1]+str(key1)
        datas[col_name] = []

        samples = self.group_data[key1].sample(n=self.n_sample,replace=True)
        if np.random.randint(10) < 1:
            print('process: ',i*100/len(self.compareDict.keys()),'%')
        for j,key2 in enumerate(self.compareDict.keys()):
            if key2 not in self.group_data.keys():
                continue
            if i<j:
                datas[col_name].append('')
                continue

            compare_samples = self.group_data[key2].sample(n=self.n_sample,replace=True)
            sim = 0
            n = 0
            for row1,row2 in zip(samples.iterrows(),compare_samples.iterrows()):
                item1 = row1[1][self.ITEM_ID] 
                item2 = row2[1][self.ITEM_ID]
                if item1!=item2 and item1 in modelb.wv.index2word and item2 in modelb.wv.index2word:
                    sim += modelb.wv.similarity(item1,item2)
                    n += 1
            if n>0:    
                datas[col_name].append(sim/n)
            else:
                datas[col_name].append('*')

        return datas
    def printSim(self,datas):
        dist_coss = pd.DataFrame(datas, index =datas.keys(), columns=datas.keys()) 
        dist_coss = dist_coss.replace(to_replace =["*",""],
                                      value =0)
#         dist_coss.T.style.background_gradient(cmap='Blues').to_excel('591_dist_0804.xlsx', engine='openpyxl') #輸出Excel
        dist_coss.T.style.background_gradient(cmap='Blues')
    def EmbeddingSim(self):
        dfg = pd.DataFrame({'key1':[str(name)+'_'+str(key1) for key1,name in self.compareDict.items()]})
        dfg['result'] = dfg.parallel_apply(self.sec_cos,axis=1)
        datas = {}
        for row in dfg['result']:
            for k,v in row.items():
                datas[k] = v
        self.printSim(datas)
