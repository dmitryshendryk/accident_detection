import os 
import csv 
import pandas 

ROOT_DIT = os.path.abspath('./')



def create_csv(train_path):
    out_put_file = 'data_list_1.csv'
    ll = os.listdir(train_path)
    print(ll)
    for det_classes in ll:
        ll_class = os.listdir(train_path + '/' + det_classes )
        train = ll_class[:len(ll_class) - int(len(ll_class) * 0.1)]
        test = ll_class[len(ll_class) - int(len(ll_class) * 0.1):]
      
        lables = ['train'] * len(train) + ['test'] * len(test)
        train = train + test
        label_class =  [det_classes] * len(train)
        size_classes = list(map(lambda x: len(os.listdir(train_path + '/' + det_classes + '/'+ x )), ll_class))
        print(len(train), train)
        print(len(test), test)

        df = pandas.DataFrame(data={"col1":lables,"col2":label_class,  "col3":train, 'col4': size_classes })
    df.to_csv('./data_list_1.csv', sep=',', index=False, header=False)