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
settings['nb_epoch'] = 5
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

# ##########################################################################  #

# tries all values in array for the parameter in the gradient_boost function
# validation results ~ 10-0.65 50-0.60 100-(0.514, 0.587) 200-(0.487, 0.575) 500-(0.442, 0.573)
# avec overfitting croissant de 0.05 à 0.13
array = [150]
model = model_problem.dreem_model(settings)
model.apply_model(array)
model.describe_self()
