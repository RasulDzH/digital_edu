import pandas as pd
df = pd.read_csv('train.csv')
df.info()
df.drop(['id', 'city', 'people_main', 'life_main', 
'last_seen', 'career_start', 'career_end', 'bdate', 'occupation_name',
'graduation','has_photo', 'has_mobile',
'followers_count', 'relation'], axis = 1, inplace = True)

def fill_sex(sex):
    if sex == "male":
        return 1
    return 0
df['sex'] = df['sex'].apply(fill_sex)

df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

def edu_status_apply('edu_status'):
    if edu_status == 'Undergraduate applicant':
        return 0
    if edu_status == 'Student (Bachelor`s)' or edu_status == 'Student (Specialist)' or edu_status == 'Student (Master`s)':
        return 1
    if edu_status == 'Alumns(Bachelor`s)' or edu status == 'Alumns (Specialisе)' or edu status == 'Alumns (Master`s)' :
        return 2
    if edu_status == 'PhD' or edu_status == 'Candidate of Sciences':
        return 3
df['education_status'] = df['education_status'].apply(edu_status_apply)
def langs_apply(langs):
    if langs.find("Русский")!= -1:
        return 0
    return 1
df['langs'] = df['langs'].apply(langs_apply)
def oc_type(ocu_type)
    if ocu_type == 'university':
        return 1
    return 0
df['occupation_type'] = df['occupation_type'].apply(oc_type)

print(df['sex'].value_counts())
print(df['education_status'].value_counts())
print(df['langs'].value_counts())

fem_student = 0
male_student


def edu_imp_sex('education_status'):
    global fem_student, male_student
    if row['education_status'] == 3:
        if row['sex'] == 1:
            fem_student += 1
        else:
            male_student += 1
    return False
df['sex'] = df.apply(edu_imp_sex, axis = 1)
print('Количество студентов-мужчин:', male_student, 'Количество студентов-женщин:', fem_student )

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsclassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x - df.drop('result', axis = 1)
y = df['result']
x train, x test, y_train, y test = train test_split(X, y, test_size = 0.25)
I

sc = StandardScaler()
x train - sc.fit_transform(x train)
x test = sc.fit transform(x test)

classifier-KNeighborsclassifier(_neighbors=5)
classifier.fit(X_train, y_train)

y_pred - classifier-predict(x test)
print(y_test)
print (y_pred)
print('Процент правильно предсказанных исходов: ', round(accuracy_ score(y_test, y_pred) *100, 2))
print ( Confusion matrix:)
print (confusion_matrix(y_test, y_pred))




