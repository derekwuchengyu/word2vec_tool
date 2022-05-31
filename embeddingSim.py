from pandarallel import pandarallel
class ClassEvaluateWord2vecModel:
    """
    使用範例:
    class w2vEvaluate(ClassEvaluateWord2vecModel):
        def __init__(self,df):
            super().__init__()
            self.df = df
            self.ITEM_ID = 'job_area'

        def searchsorted():
            pass

    eV = w2vEvaluate(dfl.head(1000))
    eV.feature_evl(jlW2v_model,'job_area',job_location,n_sample=5)
    """
    def __init__(self):
        pandarallel.initialize(progress_bar=False)
        self.df = None  #物件資料DataFrame
        self.ITEM_ID = '' #word2vec 元素

    def searchsorted(self):
        """
        數值轉編碼
        ex:
            price_range = np.array([0,500,1000,2000,3000,4000,5000,6000,9999999])
            self.df['價格編碼'] = self.df['PRICE'].apply(lambda x : price_range.searchsorted(x))
        """
        pass
        
    def feature_evl(self,model,groupby,ranges,n_sample=100):
        """ 特徵評估 """
        if len(self.df)==0:
            print("未帶入df變數!")
            return 0

        print("word2vec模型 %s 評估" %(groupby))
        data = {}
        self.group = {str(sec):g.sample(n=n_sample,replace=True) for sec, g in self.df.groupby(groupby) if len(g)>0}
        tags = []
        
        if type(ranges)==np.ndarray: # 範圍型特徵:
            stack  = lambda x : range(1,len(x))
            index  = lambda x : x
            column = lambda x : str(ranges[x-1])+'~'+str(ranges[x])

        elif type(ranges)==dict: # 類別型特徵
            stack  = lambda x : x
            index  = lambda x : list(ranges.keys()).index(x)
            column = lambda x : str(ranges[x])+str(x)
            
        else:
            return "類型錯誤!"
        
        for i in stack(ranges):
            for j in stack(ranges):
                if index(i)>=index(j):
                    tags.append(str(i)+'_'+str(j))
                    
        print("計算數量: ",len(tags))
        dataFrame = pd.DataFrame({'tag':tags})
        dataFrame = dataFrame.parallel_apply(self.func,model=model,n_sample=n_sample,axis=1)
        
        for i in stack(ranges):
            col_name = column(i)
            if col_name not in data.keys():
                data[col_name] = {}
                
            for j in stack(ranges):
                row_index = column(j)
                data[col_name][row_index] = ''
                tag = str(i)+'_'+str(j)
                value = dataFrame.query('tag==@tag')
                if len(value)>0:
                    data[col_name][row_index] = value['score'].values[0]
                        

        price_cos = pd.DataFrame(data, index =data.keys()) 
        price_cos = price_cos.replace(to_replace =["*",""],
                                      value =0)

        display(price_cos.T.style.background_gradient(cmap='Blues'))
        
        return price_cos

    def func(self,s,model,n_sample=100):
        i = str(s['tag'].split("_")[0])
        j = str(s['tag'].split("_")[1])
        try:
            samples = self.group[i]
            compare_samples = self.group[j]
        except:
            s['score'] = 'KeyError'
            return s
        
        sim = []
        for row1,row2 in zip(samples.iterrows(),compare_samples.iterrows()):
            item1 = row1[1][self.ITEM_ID] 
            item2 = row2[1][self.ITEM_ID]
            if item1!=item2: # and item1 in modelb.wv.index2word and item2 in modelb.wv.index2word:
                try:
                    sim.append(abs(model.wv.similarity(item1,item2)))
                except:
                    pass

        s['score'] = np.mean(sim) if len(sim)>0 else ''
        return s
