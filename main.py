import datetime
import model_problem
import time

# ################################ SETTINGS ################################  #
settings = dict()
# should we use the small dataset?
settings['small_dataset'] = False
# are we saving the results?
settings['saving_results'] = not settings['small_dataset']
# are we storing part of the dataset for future use?
settings['storing_small_dataset'] = False
# how to split the 50,000 training examples into training set / validation set
settings['validation_share'] = 0.2
# how many time we go through the data
settings['nb_epoch'] = 15
# name of the file where saving results
output_name = "grad_boost + CNN"

# using fft?
settings['fft'] = False

# should we use stored activations if the we use CNN + grad_boost?
settings['use_stored_activations'] = True
# should store activations this time?
settings['should_store_activations'] = False
# use only activations or compute grad_boost on activations + raw data?
settings['use_only_activations'] = True
if output_name != "grad_boost + CNN" and (settings['use_stored_activations'] or settings['use_only_activations'] or settings['should_store_activations']):
    "You're doing this wrong... don't use stored activations if you're not doing CNN + grad_boost!"

settings['output_name'] = output_name  \
              + "_" + str(datetime.date.today().day) \
              + "-" + str(datetime.date.today().month) \
              + "-" + str(datetime.date.today().year) \
              + "-" + str(datetime.datetime.now().hour) \
              + "h" + str("%02.f" % datetime.datetime.now().minute)
# what model? 'gradient_boost' / 'convolution' / 'convolution + gradient_boost'
# settings['model_option'] = 'convolution'
settings['model_option'] = 'convolution + gradient_boost'

# display evolution of training error?
settings['display'] = False

# regularization parameter (lambda)
settings['regularization_param'] = 0
# default = 0.001 (learning rate of adam = the cnn optimizer used)
settings['adam_lr'] = 0.001
# default = 0.1 (learning rate of gradient boost)
settings['grad_boost_lr'] = 0.08
# default = 1 (reducing it under 1 = regularization)
settings['grad_boost_subsample'] = 0.8
# default = None
settings['grad_boost_max_features'] = None
# gradient boost parameter (= number of estimators)
settings['grad_boost_param'] = 300
# ##########################################################################  #
start_time = time.time()
model = model_problem.dreem_model(settings)
model.apply_model()
model.describe_self()
print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
