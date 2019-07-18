def make_labels():
    path = 'static/labels.txt'
    labels = [line.rstrip('\n') for line in open(path)]
    return labels
    
def cal_table(prediction):
	import pandas as pd
	df_cal = pd.read_csv("data/nutritional_values.csv")

	df_cal = df_cal.set_index('product_name')

	return df_cal[df_cal.index == prediction].transpose()


  
  
