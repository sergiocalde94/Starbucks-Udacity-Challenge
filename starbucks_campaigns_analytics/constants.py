from pathlib import Path


# PATH_PROJECT will be called from the root of the project or from a subfolder
PATH_PROJECT = (
    Path('.') if Path('.').resolve().name == 'Starbucks-Udacity-Challenge'
    else Path('..')
)

PATH_DATA = PATH_PROJECT / 'data'
PATH_REPORTS = PATH_PROJECT / 'reports'
PATH_MODELS = PATH_PROJECT / 'models'
PATH_SUBMISSIONS = PATH_PROJECT / 'submissions'
PATH_OBJECTS = PATH_PROJECT / 'objects'
PATH_MLFLOW_TRACKING = PATH_PROJECT / 'mlruns'

TARGET = 'sales'
RANDOM_STATE = 42
