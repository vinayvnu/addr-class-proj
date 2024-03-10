import pathlib
import os
import pred_model

PACKAGE_ROOT = pathlib.Path(pred_model.__file__).resolve().parent
DATAPATH = os.Path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')
TARGET = 'Loan_Status'
FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
             'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
CAT_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'Credit_History', 'Property_Area']

# same as categorical features
FEATURES_TO_ENCODE = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'Credit_History', 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = ['CoApplicantIncome']
DROP_FEATURES = ['CoApplicantIncome']
LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']