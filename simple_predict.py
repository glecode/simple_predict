import math
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import optimize
from scipy.special import ive
from copy import deepcopy

#from functools import partial 
#from functools import partialmethod 


class Predict:    
    def __init__(self, data, window_length=100):
        self.data = data
        self.window_length = window_length
        self.window_endpoints = []
        self.full_index = self.data.copy().index
        self.data.reset_index(inplace=True, drop=True)
        print(window_length)
    
    def create_window_endpoints(self):
        window_endpoints = []
        last = self.data.shape[0]
        window_start = 0
        window_end = self.window_length - 1
        while window_end < last - 1 : 
            window_endpoints.append( (window_start, window_end) )
            window_start += 1
            window_end += 1
        self.window_endpoints = window_endpoints
        
    def predict(self , start_end ):
        slice_start = start_end[0]
        slice_end = start_end[1] + 1
        return self.data.iloc[ slice_start:slice_end , 0].mean()
        
    def rolling_full(self):
        predictions = []
        actual_values = []
        self.create_window_endpoints()
        for item in self.window_endpoints:
            predictions.append( self.predict(item) )
        self.actual_vs_pred = self.data.loc[:,self.series_name].copy().iloc[self.window_length: ].to_frame()
        self.actual_vs_pred.rename(columns={self.series_name:'actual_values'}, inplace=True)
        self.actual_vs_pred.index = self.full_index[self.window_length:]
        self.actual_vs_pred['predictions'] = predictions
        self.RMSFE = ( ( self.actual_vs_pred['actual_values'] - self.actual_vs_pred['predictions']
                       ).pow(2).sum()/self.actual_vs_pred.shape[0] )**0.5

        
class AR1(Predict):    
    def __init__(self, data, window_length=100):
        Predict.__init__(self, data, window_length)
    
    def estimate(self, start_end):
        slice_start = start_end[0]
        slice_end = start_end[1] + 1
        
        # this is y = windowdata[1:]
        y = self.data.iloc[ (1+slice_start):slice_end , 0]
        y.index = range(self.window_length - 1) # lose one observation pair due to the lag
        
        # this is ylag = windowdata[:-1]
        ylag = self.data.iloc[ slice_start:(slice_end-1) , 0]
        ylag.index = range(self.window_length - 1)
        
        ybar = y.mean()
        ylagbar = ylag.mean()        
        b_numerator = ( (ylag - ylagbar) * (y - ybar) ).sum()
        b_denominator = ( (ylag - ylagbar)**2 ).sum()
        self.b = b_numerator / b_denominator
        self.a = ybar - self.b * ylagbar

        
    def predict(self , start_end ):
        self.estimate(start_end)
        return self.a + self.b * self.data.iloc[ start_end[1] , 0]
    
    
class CIR(Predict):
    def __init__(self, data, window_length=100, delta=1/12, correction=False, ML=False, correctionparams_file = None):
        Predict.__init__(self, data, window_length)
        self.delta = delta
        self.correction=correction
        self.kappas_list = []
        self.ML=ML
        self.correctionparams_file = correctionparams_file
        self.series_name = data.columns[0]
    
    def estimate(self, start_end):
        slice_start = start_end[0]
        slice_end = start_end[1] + 1
        
        # this is y = windowdata[1:]
        y = self.data.iloc[ (1+slice_start):slice_end , 0]
        y.index = range(self.window_length - 1) # lose one observation pair due to the lag
        
        # this is ylag = windowdata[:-1]
        ylag = self.data.iloc[ slice_start:(slice_end-1) , 0]
        ylag.index = range(self.window_length - 1)
        
        b1_numerator = ( ( 1/(self.window_length-1)**2 ) * y.sum() * (ylag**(-1)).sum() 
                            - ( 1/(self.window_length-1) ) * ( y * (ylag**(-1)) ).sum() )
        b1_denominator = ( 1/(self.window_length-1)**2 )* ylag.sum() * (ylag**(-1)).sum() - 1
        b1 = b1_numerator/b1_denominator        
        b2_numerator = ( 1/(self.window_length-1) ) * ( y * (ylag**(-1)) ).sum() - b1          
        b2_denominator = (1 - b1) * ( 1/(self.window_length-1) ) * (ylag**(-1)).sum()
        b2 = b2_numerator / b2_denominator             
        b3 = ( 1/(self.window_length-1) ) * ( ( ( y - b1*ylag - b2*(1-b1) )**2 ) * (ylag**(-1)) ).sum()   
        
        self.kappa_hat = -(self.delta)**(-1) * math.log(b1)
        self.sigmasq_hat = 2 * self.kappa_hat * b3 / ( 1 - b1**2 )
        self.alpha_hat = b2
        
        if self.ML == True:
            self.MLE(y, ylag)
        else:
            pass
        
        if self.correction == True:
            self.make_correction()
        else:
            pass        
        self.kappas_list.append(self.kappa_hat)
    
    def make_correction(self):
        if self.correctionparams_file == None:
            lower = -1
            for item in [(a , a+60) for a in range(60,481,60)]:
                if self.window_length >= item[0] and self.window_length < item[1] :
                    lower = item[0]
                else:
                    pass
                if self.window_length >= 360 and self.window_length < 420 :
                    lower = 300
            z_df = pd.read_csv(f"Results_Prediction/from_Barkla/bestperformancepar_{lower}_{lower+60}.dat", 
                               header=None)
        else:
            z_df = pd.read_csv(self.correctionparams_file, header=None)
            

        z = z_df.values
        z = z.reshape( z.shape[0])
        
        numberofpowers = 10
        
        kappa_powers_list = []
        for item in range(numberofpowers + 1):
            kappa_powers_list.append( self.kappa_hat**item )
        
        kappa_powers = np.array(kappa_powers_list)
        kappa_powers = kappa_powers.reshape( kappa_powers.shape[0] )
        
        cT = self.window_length - 1
        index_factor = int(z.shape[0]/4) # i.e. 10
        
        correction = (1.0/cT) * ( np.dot( z[0:index_factor] , kappa_powers[0:numberofpowers] ) )  \
                          / (1 + np.dot( z[index_factor:(2*index_factor)] , \
                                      kappa_powers[1:(numberofpowers+1)]   ) ) \
                    + (1.0/cT**2) * ( np.dot( z[(2*index_factor):(3*index_factor)], \
                                          kappa_powers[0:numberofpowers]  ) ) \
                      / ( 1 + np.dot( z[(3*index_factor):(4*index_factor)] , \
                                    kappa_powers[1:(numberofpowers+1)] ))
        
        self.kappa_hat = self.kappa_hat + correction

    @staticmethod 
    def likelihood(params, y, ylag):
        y = y.to_numpy(copy=True)
        ylag = ylag.to_numpy(copy=True)

        h = 1/12
        c = 2 * params[1] / ( params[2] * ( 1 - math.exp( -params[1] * h ) ) )
        u = c * ylag * math.exp(-params[1] * h )
        v = c * y
        q = 2 * params[1] * params[0] / params[2] - 1

        zarg = 2 * np.power( u * v , 0.5 )
        eIq = ive(q, zarg )

        returnvalue = - np.sum(np.log( c) -u-v + np.log( np.power( np.divide(v , u) , q/2 ) )
                               + np.absolute(zarg) + np.log( eIq )  ) 
        return returnvalue
        
    def MLE(self, y, ylag): 
        #params0 = np.array( [ self.alpha_hat , self.kappa_hat , self.sigmasq_hat ] )
        params0 = np.array( [ 0.05,0.1,0.01 ] )
        result = minimize(self.likelihood, params0, args= (y, ylag), method='Nelder-Mead', tol=1e-6)
        #result = minimize(self.likelihood, params0, args= (y, ylag), method='Nelder-Mead')
        #result = minimize(self.likelihood, params0, args= (y, ylag), method='BFGS' )  
        
        rranges = (slice(0.01, 0.2, 0.005), slice(0.01, 0.4, 0.005) , slice(0.01, 0.2, 0.005) )
        if result.x[1]==0.1 or (result.x[1] > 0.5) or ( result.x[1] is None ):
            resbrute = optimize.brute(self.likelihood, rranges, args=(y, ylag), full_output=True,
                           finish=optimize.fmin)            
            self.alpha_hat, self.kappa_hat, self.sigmasq_hat = tuple(resbrute[0])
        else:
            self.alpha_hat, self.kappa_hat, self.sigmasq_hat = tuple(result.x)
        
    def predict(self , start_end ):
        self.estimate(start_end)
        return self.alpha_hat + ( self.data.iloc[ start_end[1] , 0 ] - self.alpha_hat ) * math.exp(-self.kappa_hat)

    
def compare_windows(data, prediction_approaches, window_lengths, results_out = False, plot_out = None, plot_display = True,  **kwargs):    #prediction methods = {"original": object_original_dict, "improved": object_improved_dict}   object_original_dict must include a key sp_type
    RMSFEs = {}
    for key in prediction_approaches:
        RMSFEs[key] = []
        
    for window_length in window_lengths:
        for approach in prediction_approaches:
            sp_type = prediction_approaches[approach]['sp_type']
            approach_copy = deepcopy(prediction_approaches[approach])
            del approach_copy['sp_type']
            approach_copy['window_length'] = window_length
            print(globals()[sp_type] )
            print(approach_copy)
            predict_object = globals()[sp_type]( data, approach_copy )
            
            #(https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string)
            #predict_object.rolling_full()
            rolling = getattr(predict_object, 'rolling_full')
            rolling()
            
            RMSFEs[key].append(predict_object.RMSFE)
    
    window_lengths = pd.Series(window_lengths)
    RMSFEs = pd.DataFrame(RMSFEs, index=window_lengths)
    
    if results_out != None:
        RMSFEs.to_csv(results_out , index=False)
    
    if plot_out != None:
        plt.savefig(plot_out)
    
    if plot_display == True:
        RMSFEs.plot(kwargs)
        
    return RMSFEs
    