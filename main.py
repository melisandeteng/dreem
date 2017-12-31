import datetime
import model_problem

# ################################ SETTINGS ################################  #
settings = dict()
# should we use the small dataset?
settings['small_dataset'] = False
# are we saving the results?
settings['saving_results'] = not settings['small_dataset']
# are we storing part of the dataset for future use?
settings['storing_small_dataset'] = False
# how to split the 50,000 training examples into training set / validation set
settings['validation_share'] = 0.5
# how many time we go through the data
settings['nb_epoch'] = 7
# name of the file where saving results
output_name = "grad_boost + CNN"
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
settings['regularization_param'] = 0.001
# default = 0.001 (learning rate of adam = the cnn optimizer used)
settings['adam_lr'] = 0.001
# default = 0.1 (learning rate of gradient boost)
settings['grad_boost_lr'] = 0.01
# default = 1 (reducing it under 1 = regularization)
settings['grad_boost_subsample'] = 0.5
# default = None
settings['grad_boost_max_features'] = 2
# gradient boost parameter (= number of estimators)
settings['grad_boost_param'] = 3000
# ##########################################################################  #

model = model_problem.dreem_model(settings)
model.apply_model()
model.describe_self()
