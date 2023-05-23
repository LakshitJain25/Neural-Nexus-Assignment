import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


#taking path for the input file
input_path = input("Give complete path of the test csv file: ")

#reading the test csv
test_df = pd.read_csv(input_path)
test_df = test_df.fillna('not mentioned')
#dropping unncessary cols
drop_cols = ['Unnamed: 0','Agent_id','lead_id']
for col in drop_cols:
  if col in test_df.columns:
    test_df = test_df.drop(col,axis=1)

if 'status' in test_df.columns:
    test_df_won = test_df[test_df['status'] == 'WON']
    test_df_lost = test_df[test_df['status'] == 'LOST']
    test_df_wl = pd.concat([test_df_won,test_df_lost]).sample(frac=1)
    test_df = test_df_wl.copy().reset_index(drop=True)


#Sub categories of lost reason
lost_reason = {
    'admission' : ['Cross sell','Looking for loan','Looking for Scholarship','Guarantor issue','Underage student','Visa/admission denied','Looking for admission', 'Not a student'],
    'no_response' : ['Not responding', "Didn't respond in time"],
    'junk' : ['Junk lead', 'Inadequate details', 'Just Enquiring', 'Junk lead/ Just Enquiring', 'Lead issue', 'Repeat lead'],
    'supply' : ['No supply', 'Supply issue'],
    'booked_na' : ['Booked with manager', 'Booked with competitor', 'Booked on campus accommodation', 'Low availability'],
    'location' : ['Distance issue', 'Not serving in that region'],
    'stay_accomodation' : ['Wants private accommodation', 'Short stay', 'Semester stay'],
    'budget' : ['Low budget'],
    'not_interested' : ['Not interested', 'Not going to university']
}
# categorizing lost reason values
def get_reason(x):
    for key,vals in lost_reason.items():
        if x in vals:
            return key
    return 'not mentioned'

test_df['lost_reason'] = test_df['lost_reason'].apply(lambda x : get_reason(x))
# defining dictionary for the coded columns SORRY FOR THE LONG MESS
code_cols_dict = {'source': ['7aae3e886e89fc1187a5c47d6cea1c22998ee610ade1f2b7c51be879f0c37ca8',
  '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0',
  '7bcfca0e9d73371699b0adbf1c691e02108fa64f02f4dbe24a0217f139a9b916',
  'ba2d0a29556ac20f86f45e4543c0825428cba33fd7a9eaa29e3f169d2ca43b2b',
  '146fb4ecbe78caa24102bbaac12e2559a8c8e32fb116d36e3553a709fce40549',
  'd684761c17c11590f6e2525b48141cb2c0c6f2be5df4e229dae06e64c5c41b64',
  '9ba9134a91cfc6b52ac8d480e9ad37896ca4ac216e2d795fdb7d75a63d6c60af',
  '9fd09dc33545f9cc19b81ebd0b98c4fd8c66ed1e34de89f4c9a81e6b26dc0d54',
  'b2b2a0ecb072ed25f1844a3325a810b85689bcc785ddb40dbdcd50a237e40831'],
 'source_country': ['e09e10e67812e9d236ad900e5d46b4308fc62f5d69446a9750aa698e797e9c96',
  '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0',
  '8da82000ef9c4468ba47362a924b895e40662fed846942a1870a674e5c6d1fc2',
  'e39b0c5e88f19053d3a917669bc9d60729f351e064ca0c94f5dc73f4e676333f',
  '38fe804a1f9ec032ad876bd7192c1f706e1402831e163cf72702b1f451f40cac',
  '0207c236c5ad89235d814b1e1807f6fdc1930810439489f76600e1672338e42b',
  'b936ee09e20b3b2234907cde349cda1c1a5327c4a486bf27cee28623bb25bb12',
  'c2863266ba318106a050f6f52c0a0e5ee19bdbacc19c0965979ae9f2b22354c6',
  'e28406d05650a1fab7eb01a80ef73a2e3460a214fe48601b63f93d37a65ab966'],
 'source_city': ['9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0',
  'ecc0e7dc084f141b29479058967d0bc07dee25d9690a98ee4e6fdad5168274d7',
  'b384863fa1f6e091164b74219726eee0b9dd47776b91a4dc71fd0516630d21ec',
  '9f2ba6981e36ea0fca6c16f42e6413d788d2a7499b8b7f418c5d76f12d191f02',
  '7d1550b844ff586a6023216c06263105eed0a849a2a1f69bb8862ab288d8cdab',
  '810c069040f6a9b16fdf976a901755508a87cb0224b169d4b5d803357e647947',
  '4fa64bd55d5c0c1f83015952b4b9500cb099dd0b1b04647f1a7c3c5708295b6e',
  '282f96b099630502f8bb5033849c69982dd147015853061ed938c2a254fd8dcf',
  '5e02dadaa7e4cd29809e2f8a115f1e5e9c01b29813a1fb9707218804218da32a'],
 'utm_source': ['bbdefa2950f49882f295b1285d4fa9dec45fc4144bfb07ee6acc68762d12c2e3',
  '7f3fa48ca885678134842fa7456f3ece53a97f843b610185d900ac4e467c7490',
  '3d59f7548e1af2151b64135003ce63c0a484c26b9b8b166a7b1c1805ec34b00a',
  '3c77f261a156a5308fee53720276395ef78d2e7367e4225a3d3d93f4accd1dd3',
  'd15690f08a575024650b01ffac892cfd2b93e6c57c140f1b6d9e47753cabd579',
  '9ac56ef275bb33f0f931abd846e53e845a80af8a549100741e928b8b2abd56a7',
  'ec8202b6f9fb16f9e26b66367afa4e037752f3c09a18cefab426165e06a424b1',
  '4f48c17d2a97a7461a12d0d07336f808e70a1248a7082dbb103047f54090c158',
  '44574c4ba2ea74ad4bf1e184133cdbf4e7390a3690beff6a7364511a70ec208e'],
 'utm_medium': ['09076eb7665d1fb9389c7c4517fee0b00e43092eb34821b09b5730c41ebcc50c',
  '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0',
  '09bc8f0bb77bdddeb98527a39e995c2c605166399c178cb296992abab8415d5b',
  '69f81619d0ef92cbb165a44e76f4f0c284d2aa83c318fa7fb632207f68215647',
  '268ad70eb5bc4737a2ae28162cbca30118cc94520e49ef1ac5f72c85d3f2caa9',
  '82244417f956ac7c599f191593f7e441a4fafa20a4158fd52e154f1dc4c8ed92',
  '2bc856d8b9bd2e94d0aacf13e9dac47f9259f8dcf56061896f28dda46081c393',
  'abb8e2badd5b6265c3237170cc599257a4f566706715d2e8ed911caf07185447',
  '08510d8a07a19e4b995447e77c3b1a40c6f21838ecae77448a01d98312a00b6c'],
 'des_city': ['ecc0e7dc084f141b29479058967d0bc07dee25d9690a98ee4e6fdad5168274d7',
  '810c069040f6a9b16fdf976a901755508a87cb0224b169d4b5d803357e647947',
  '11ab03a1a8c367191355c152f39fe28cae5e426fce49efb320230ca4ae3f97a1',
  '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0',
  '8593b9db65212160979d946950601c26622a219c80f1f122679eb69ec5b96600',
  '6a5ed83658ef85afc77709cc16ede854ad98e3c2ad8b076e10dcbcd6e5096271',
  '5e02dadaa7e4cd29809e2f8a115f1e5e9c01b29813a1fb9707218804218da32a',
  '9b8cc3c63cdf447e463c11544924bf027945cbd29675f77955bb36364356c14e',
  'b2586a6cef5690b74e9fb425f95f8fb3f1e18a4cdc3225eb2f53534ec3602aee'],
 'des_country': ['8d23a6e37e0a6431a8f1b43a91026dcff51170a89a6512ff098eaa56a4d5fb19',
  '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0',
  '02bf1dfa9a0365a86223f0e4ac9eaa0517b06b2e9273790f719feda656a247ae',
  '80db4ccdca106d37b920206331fcfe3e9e50a9e763d89b54ce3ad5ac8cf30f03',
  'be55ef3f4c4e6c2d9c2afe2a33ac90ad0f50d4de7f9163999877e2a9ca5a54f8',
  '7a1ca4ef7515f7276bae7230545829c27810c9d9e98ab2c06066bee6270d5153',
  'c1ef40ce0484c698eb4bd27fe56c1e7b68d74f9780ed674210d0e5013dae45e9',
  '5a9cf672c8be6b5ab9546a2fb49b06dd81a4e364c86ed023898c49d9bb0605dc',
  '49dca65f362fee401292ed7ada96f96295eab1e589c52e4e66bf4aedda715fdd']}

code_cols = ['source','source_country','source_city','utm_source','utm_medium','des_city','des_country']
def get_col(x,col,col_dict):
    if x in col_dict[col]:
        return x
    return "not mentioned"

for col in code_cols:
    test_df[col] = test_df[col].apply(lambda x : get_col(x,col,code_cols_dict))

# Categorizing rooms
rooms = ['Ensuite','Studio','Entire Place']
def get_room_type(x):
  if x in rooms:
    return x
  return "not mentioned"
test_df['room_type'] = test_df['room_type'].apply(lambda x : get_room_type(x))


def get_nums(x,month):
  numbers = re.findall(r'[0-9]+', x)
  for i in range(len(numbers)):
      numbers[i] = int(numbers[i])
  if month:
    for i in range(len(numbers)):
      numbers[i] = 4 * numbers[i]
  return numbers

weeks = ['pw','ppw','week','weeks','weekly']
months = ['pm','ppm','month','months','monthly','pcm']
def get_month(x):
  for word in x:
      if word in months:
        return True
  return False

def get_budget(x):
    x = str(x).lower()
    x = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", x)
    alphabets = re.findall(r'[a-zA-Z]+', x)
    month = get_month(alphabets)
    nums = get_nums(x,month)
  # print(alphabets,nums,month)
    if len(nums) == 0:
        return [-1,-1]
    elif len(nums) == 1:
        return [nums[0],nums[0]]
    elif len(nums) == 2:
        return [nums[0],nums[1]]
    elif len(nums) == 3 or len(nums) == 4:
        return [nums[1],nums[2]]
    
test_df['budget'] = test_df['budget'].fillna("not_mentioned")
mins = []
maxs = []
for x in test_df['budget']:
  got = get_budget(x)
  if got[0] > 10000 or got[1] > 10000:
     got = [-1,-1]
  mins.append(got[0])
  maxs.append(got[1])
test_df['min_budget'] = mins
test_df['max_budget'] = maxs
drop_cols = ['budget']
test_df = test_df.drop(drop_cols,axis=1)

years = ['year','years']
months = ['month','months','mnth']
week = ['weeks','week']
days = ['day','days']
sem = ["sem","course"]

def getnum(n):
  a = 0;
  for x in n:
    a += int(x)
  return a // len(n)

def get_duration(x):
  x = x.lower()
  alphabets = re.findall(r'[a-zA-Z]+', x)
  numbers = re.findall(r'[0-9]+', x)
  if len(alphabets) == 0:
    if len(numbers) == 1:
      if(numbers[0] == '1'):
        return 52
      elif len(numbers[0]) == 1:
        return 4 * int(numbers[0])
      elif len(numbers[0]) == 2:
        return int(numbers[0])
      elif len(numbers[0]) == 3:
        return int(numbers[0]) // 7
      return getnum(numbers)
      

  elif len(numbers) > 0 and len(alphabets) > 0:
    for a in alphabets:
      a = a.lower()
      if a in days:
        return getnum(numbers) // 7
      elif a in weeks:
        return getnum(numbers)
      elif a in months:
        return 4 * getnum(numbers)
      elif a in years:
        return 52 * getnum(numbers)
      return getnum(numbers)

  else:
    for a in alphabets:
      a = a.lower()
      if "sem" in a or "course" in a:
        return 24
      elif a in months:
        return 4 
      elif a in years:
        return 52
  return -1

test_df['duration'].fillna("not mentioned",inplace=True)
dur_list = test_df['duration'].apply(lambda x : get_duration(x))
dur_list = [-1 if x > 10000 else x for x in dur_list]
test_df['duration'] = dur_list

if 'status' in test_df.columns:
      status_map = {
      'WON' : 1,
      "LOST" : 0
  }
test_df['status'] = test_df['status'].map(status_map)


X = test_df.drop('status',axis=1)
y = test_df['status']

cat_cols = ['lost_reason', 'source', 'source_city', 'source_country','utm_source', 'utm_medium', 'des_city', 'des_country', 'room_type']
le_dict = {}
for col in cat_cols:
  le = LabelEncoder()
  encoded = le.fit_transform(X[col])
  X[col] = encoded
  le_dict[col] = le

model_path = input("Give complete path of the downloaded model: ")

loaded_model = pickle.load(open(model_path, 'rb'))
preds_prob = loaded_model.predict_proba(X)
preds = loaded_model.predict(X)

revmap = {
    1 : 'WON',
    0 : 'LOST'
}

actual_preds = [[revmap[a],b[1]*100] for a,b in zip(preds,preds_prob)]

result_dict = {
    "Lead Score" : [],
    "Predicted Label":[],
    "Actual Label": [revmap[x] for x in y]
}

for a in actual_preds:
  result_dict['Predicted Label'].append(a[0])
  result_dict['Lead Score'].append(a[1])

final_df = pd.DataFrame(result_dict)
final_df.to_csv("results.csv",index=False)

def compute_metrics(y_true,y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


mets = compute_metrics(y,preds)
print("\n\nF1-SCORE IS : ",mets['f1'],'\n')
print(final_df)
