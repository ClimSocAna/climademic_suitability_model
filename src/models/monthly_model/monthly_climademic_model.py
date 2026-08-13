# models/libsvm_ocsvm.py
from libsvm.svmutil import (
    svm_problem,
    svm_parameter,
    svm_train,
    svm_predict,
    svm_save_model,
    svm_load_model,
)
""" 
docs from here. https://docs.ros.org/en/fuerte/api/libsvm3/html/namespacelibsvm_1_1svmutil.html
Train an SVM model from data (y, x) or an svm_problem prob using
'options' or an svm_parameter param. 
If '-v' is specified in 'options' (i.e., cross validation)
either accuracy (ACC) or mean-squared error (MSE) is returned.
'options':
    -s svm_type : set type of SVM (default 0)
        0 -- C-SVC
        1 -- nu-SVC
        2 -- one-class SVM
        3 -- epsilon-SVR
        4 -- nu-SVR
    -t kernel_type : set type of kernel function (default 2)
        0 -- linear: u'*v
        1 -- polynomial: (gamma*u'*v + coef0)^degree
        2 -- radial basis function: exp(-gamma*|u-v|^2)
        3 -- sigmoid: tanh(gamma*u'*v + coef0)
        4 -- precomputed kernel (kernel values in training_set_file)
    -d degree : set degree in kernel function (default 3)
    -g gamma : set gamma in kernel function (default 1/num_features)
    -r coef0 : set coef0 in kernel function (default 0)
    -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
    -m cachesize : set cache memory size in MB (default 100)
    -e epsilon : set tolerance of termination criterion (default 0.001)
    -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0).for one-class SVM only 0 is supported.
    -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
    -v n: n-fold cross validation mode
    -q : quiet mode (no outputs) """

class ClimademicMonthlyModel:
    def __init__(self, params="-s 2 -t 2 -g 0.03 -n 0.03 -b 1 -q"):
        self.params = params
        self.model = None

    def train(self, X_train, y_train):
        problem = svm_problem(y_train, X_train)
        parameters = svm_parameter(self.params)
        self.model = svm_train(problem, parameters)
        return self.model

    def predict(self, X, probability=True):
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        options = "-b 1 -q" if probability else "-b 0 -q"
        labels, accuracy, values = svm_predict([], X, self.model, options)

        return labels, accuracy, values

    def save(self, path):
        if self.model is None:
            raise ValueError("No model to save.")
        svm_save_model(str(path), self.model)

    def load(self, path):
        self.model = svm_load_model(str(path))