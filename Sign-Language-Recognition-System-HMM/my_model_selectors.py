import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    
    The term −2 log L decreases with increasing model complexity (more parameters), whereas the penalties 2p or
    p log N increase with increasing complexity. The BIC applies a larger penalty when N > e 2 = 7.4.
    
    Model selection: The lower the BIC value the better the model.

    
    N (numFeatures) = Length of the time series of the observartions
    p(numParameters) = Initial state occupation probabilities + Transition probabilities + Emission probabilities - 1
    
    Initial state occupation probabilities = number of States (numStates)
    Transition probabilities = numStates * (numStates - 1)
    Emission probabilities  = numStates*numFeatures*2
    
    Hence,
    numParameters = numStates + (numStates * (numStates - 1)) + numStates*numFeatures*2 -1
      = numStates**2 + 2 * numStates * numFeatures - 1
      
    Substitue value in BIC Equation
    BIC = -2 * log L + numParameters * log (numFeatures)
    
    References: 
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
    https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12   
    
    
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_model = {}
        for numStates in  range (self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(numStates)  
                log_likelihood = hmm_model.score(self.X,self.lengths)
                numFeatures = sum(self.lengths)
                
                # Read Comments above to understand the derivation of numParameters
                numParameters =  numStates**2 + (2 * numStates * numFeatures) - 1
            
                bic_score =  -2*log_likelihood + numParameters * np.log(numFeatures)
            
                # Collect BIC valuess Calculated for all models with different hidden states
                best_model[hmm_model] = bic_score
            except Exception:
                pass   
         
        #Return the model with the lowest BIC value.  (Read Comments above to understand Why?)
        return min(best_model, key=best_model.get) if len(best_model) > 0 else None
            


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    Tranlating the above equation into the Projects terms:
        
    log(P(X(i))  = Log likelihood of the Current Word
    1/(M-1)SUM(log(P(X(all but i))  = Average Log likelihood of all the other words.
    
    Hence
    
    DIC = logL (Current Word)  - mean (logL (of all the other words))
    
    
    DIC scores the discriminant ability of a training set for one word against competing words. 
    Instead of a penalty term for complexity, it provides a penalty if model liklihoods for non-matching words 
    are too similar to model likelihoods for the correct word in the word set.
    
    In Short, The DIC is the sum of two terms. 
    
    The first term is a difference between the likelihood of the data,log(P(X(i)), 
    and the average of anti-likelihood terms, 1/(M-1)SUM(log(P(X(all but i)), where the anti-likelihood of the
    data X against model M is a likelihood-like quantity in which the data and the model belong to competing categories.
    
    The second term is zero when all data sets are of the same size.
    
    Model Selection:
        
    Discriminant Factor Criterion (DIC) is the difference between the evidence of the model, 
    given the corresponding data set, and the average over anti-evidences of the model. By choosing the model which maximizes 
    the evidence, and minimize the antievidences, the result is the best generative model for the correct class and 
    the worst generative model for the competitive classes
    
    References:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    
    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        
        def mean_otherWord_logLikelihoods(model, current_word):
            otherwords_logL = []
            for word in self.hwords.keys():
                if word != current_word:
                    sequencesX, lengthsX = self.hwords[word]
                    otherwords_logL.append(model.score(sequencesX, lengthsX))
            return np.mean(otherwords_logL)

        # TODO implement model selection based on DIC scores
        models_and_LogLs = {}
        for numStates in  range (self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(numStates)  
                log_likelihood = hmm_model.score(self.X,self.lengths)
                
                # Collect BIC valuess Calculated for all models with different hidden states
                models_and_LogLs[hmm_model] = log_likelihood
            except Exception:
                pass  
            
        best_model={}
        for model,logL in models_and_LogLs.items():
            averageLogL = mean_otherWord_logLikelihoods(model, self.this_word)
            dic_score = logL - averageLogL
            best_model[model] = dic_score
            
         
        #Return the model with the highest DIC value. (Read Comments above to understand Why?)
        return max(best_model, key=best_model.get) if len(best_model) > 0 else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        K-Fold cross validation** provides a proxy method of finding the best model to use on "unseen data". 
        The  data is divided into k randomly assigned subsets and everytime one of the k-subsets is held out as the testing set.
        Training is done on remaining k-1 subsets and performance is measured on the test set. 
        The average error across all k-trials is computed. 
          1) As k increases, the variance of the resulting estimate is reduced.
          2) The main advantage of this technique is it avoids overfitting
                
                
        The number of folds used in this code is 3.
        If the length of sequences in < 3 then K fold validation is not used.
    ''' 
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Use the range min_n_components and max_n_components for hidden states, 
        # Test the model for with different hidden states.
        num_folds = min(3,len(self.sequences))
        kFoldCV = KFold(n_splits = num_folds)
        
        
        best_model = {}
        for hidden_states in  range (self.min_n_components, self.max_n_components + 1):
             try:
                 if num_folds > 2 :
                     test_log_likelihood = {}
                     for cv_train_idx, cv_test_idx in kFoldCV.split(self.sequences):
                         self.X, self.lengths = combine_sequences(cv_train_idx,self.sequences)
                         hmm_model = self.base_model(hidden_states)                 
                         
                         test_X, test_length = combine_sequences(cv_test_idx, self.sequences)
                         
                        # Capture the log likelihood on Test set for every fold of training set.
                         log_likelihood = hmm_model.score(test_X,test_length)
                         test_log_likelihood[hmm_model] = log_likelihood
                     
                     #Capture the Model which returns the best log likelihood on the set set
                     best_model_fold = max(test_log_likelihood, key=test_log_likelihood.get)
                     log_likelihood =  test_log_likelihood[best_model_fold]                    
                 else:
                     best_model_fold = self.base_model(hidden_states)  
                     log_likelihood = best_model_fold.score(self.X,self.lengths)
                 
                 # Collect log likelihoods returned by models for all hidden states
                 best_model[best_model_fold] = log_likelihood
                 
             except Exception:
                pass
        #Return the model with best log likelihood.
        return max(best_model, key=best_model.get)
            

    
    
            
            
            
            
            
            
                 
                 
            
        
