# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
# model = 'NeuMF-pre' # 아래 pth 파일이 없어서 에러 발생
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

main_path = '../data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
gs_NeuMF_model_path = model_path + 'NeuMF-end.pth'
