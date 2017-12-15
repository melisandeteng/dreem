import datetime
import model_problem

# ################################ SETTINGS ################################  #

# should we use the small dataset?
small_dataset = False
# are we saving the results?
saving_results = False
# are we storing part of the dataset for future use?
storing_small_dataset = False
# name of the file where saving results
output_name = "convnet"
output_name = output_name  \
              + "_" + str(datetime.date.today().day) \
              + "-" + str(datetime.date.today().month) \
              + "-" + str(datetime.date.today().year) \
              + "-" + str(datetime.datetime.now().hour) \
              + "h" + str("%02.f" % datetime.datetime.now().minute)

# ##########################################################################  #

array = [10, 50, 100, 200, 500]
results = [[]]
for i in range(0, len(array)):
    res_train, res_valid = model_problem.gradient_boost(small_dataset, saving_results, storing_small_dataset, output_name, array[i])
    results += [[array[i],res_train, res_valid]]
print(results)
