



import os


models = ['pumat21','pumat42','plasimt21', 'plasimt42']
train_years_list = [30,100]

for model in models:
    for train_years in train_years_list:
        print(model,train_years)
        os.system('python puma_plasim_make_climate_sims_v7.py '+str(model)+' '+str(train_years))

