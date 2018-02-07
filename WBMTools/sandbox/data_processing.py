import pylab as pl
import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def survey_results(dir_survey, survey):
	""" This function gets the survey scales and put it in a data frame named survey_scales.
	This function gets some survey info: person_id, person_ip, start and end time and put it in a data frame named
	survey_info.
	Parameters
	----------
	dir_survey: string
	  the file directory.
	survey: int
	  to get scales indexes
	Returns
	-------
	survey_info: dataframe
		indexes: person_id (int), person_ip (int), start_time (string), end_time (string)
	survey_scales: dataframe
		indexes: max (int), regret (int), neuro (int), neg_affect (int), self_repr (int), satisf (int), dec_diff (int),
		alt_search (int)
	"""
	survey_info = pd.read_csv(dir_survey, sep=';', header=0, index_col=0, skiprows=2)

	return survey_info


def filter_subjects(survey_info):
	birthday = survey_info['s0_age'].tolist()
	ages = []

	for i in birthday:
		b = dt.datetime.strptime(i, '%d/%m/%Y %H:%M')
		t = dt.date.today()
		age = t.year - b.year
		ages.append(age)

	fake_subj = pl.find(pl.array(ages) < 18)
	survey_info = survey_info.drop(survey_info.index[fake_subj])

	return survey_info, pl.array(ages)[pl.array(ages) >= 18]


def statistical_population(survey_info, ages, survey):
	sex = survey_info['s0_sex'].tolist()
	scale = survey_info['s0_max'].tolist()

	mean_age = pl.mean(ages)
	std_age = pl.std(ages)
	fig, ax = plt.subplots()
	counts, bins, patches = ax.hist(ages)

	# Set the ticks to be at the edges of the bins.
	ax.set_xticks(bins)
	pl.title('Survey ' + str(survey))
	# Label the raw counts and the percentages below the x-axis...
	bin_centers = 0.5 * np.diff(bins) + bins[:-1]
	for count, x in zip(counts, bin_centers):
		# Label the percentages
		percent = '%0.0f%%' % (100 * float(count) / counts.sum())
		pl.text(x, count+0.5, str(percent), fontsize=12)
	pl.text(45, 65, "mean: " + str(mean_age), fontsize=12)
	pl.text(45, 60, "std: " + str(std_age), fontsize=12)

	# Give ourselves some more room at the bottom of the plot
	plt.subplots_adjust(bottom=0.15)
	plt.show()
	pl.title('Survey ' + str(survey))
	male = pl.find(pl.array(sex) == 2)
	female = pl.find(pl.array(sex) == 1)
	cs = mpl.cm.Set1(np.arange(2) / 2.)
	pl.pie([len(male)*100/len(sex), len(female)*100/len(sex)], labels=['Male', 'Female'], autopct='%1f%%', shadow=True, startangle=90, colors=['#0489B1', '#FACC2E'])
	pl.show()

	sns.set(style="whitegrid", color_codes=True)
	new_dataframe = pd.DataFrame(columns=('scale', 'value'))
	for i in ['s0_max', 's0_regret', 's0_neuro', 's0_neg_affect', 's0_self_repr', 's0_satisf', 's0_dec_diff', 's0_alt_search']:
		for j in range(len(survey_info[i].tolist())):
			new_dataframe.loc[-1] = [i, survey_info[i].tolist()[j]]
			new_dataframe.index = new_dataframe.index + 1  # shifting index

	sns.violinplot(x="scale", y="value", data=new_dataframe, inner=None)
	sns.swarmplot(x="scale", y="value", data=new_dataframe,  color="gray", edgecolor="gray")
	pl.show()




def get_person_id(mouse_file, dataset_path, server_info):
	""" This function gets the person ID from mouse movements file.
	This step is important to relate with the survey results.
	Parameters
	----------
	mouse_file: file
	  file to analyse.
	dataset_path: str
	  to localize id.
	server_info: dataframe
	Returns
	-------
	server_info: dataframe
		index: person_ip (string)
	"""

	mouse_file = open(mouse_file, 'r')
	i_start = mouse_file.name.index(dataset_path) + len(dataset_path)
	i_end = i_start + mouse_file.name[i_start:].index("_")
	server_info['person_ip'] = [mouse_file.name[i_start:i_end]]
	return server_info


def get_survey_id(mouse_file, server_info):
	""" This function gets the survey ID from mouse movements file.
	This step is important to select the survey to analyse.
	Parameters
	----------
	mouse_file: file
	  file to analyse.
	server_info: string
	  to use index person_ip
	  the number of the survey appears after the person ip (eg 85.7.249.186_465687)
	Returns
	-------
	server_info: dataframe
	  new index: survey_id (string)
	"""
	person_id = server_info['person_ip'][0]
	mouse_file = open(mouse_file, 'r')
	i_start = mouse_file.name.index(person_id) + len(person_id) + 1
	i_end = i_start + 6
	server_info['survey_id'] = [mouse_file.name[i_start:i_end]]
	return server_info


def get_step(mouse_file, server_info):
	""" This function gets the survey's step from mouse movements file.
	This step is important to join step in the same group.
	Parameters
	----------
	mouse_file: file
	  file to analyse.
	server_info: dataframe
	  to use survey_id
	  the step number appears after the survey id (eg 465687_1)
	Returns
	-------
	server_info: dataframe
	  new indexes: step (string), time (string)
	"""

	survey_id = server_info['survey_id'][0]
	mouse_file = open(mouse_file, 'r')
	i_start = 10+mouse_file.name[10:].index(survey_id) + len(survey_id)
	if mouse_file.name[i_start] == "_":  # case 465687_1
		i_start += 1
		i_end = i_start + 1
	else:  # case 4656871
		i_end = i_start + 1
	server_info['step'] = [mouse_file.name[i_start:i_end]]
	server_info['time'] = [mouse_file.name[i_end + 1:i_end + 1 + mouse_file.name[i_end + 1:].index("_")]]
	return server_info


def get_subj_pos(survey_info, server_info):
	""" This function returns the subject id based on his ip and time of answer.
	Parameters
	----------
	survey_info: dataframe
	  to use start_time, end_time, person_ip, person_id
	server_info: dataframe
	  to use time, person_ip
	Returns
	-------
	subj_pos:int
	  position in survey of subject
	"""
	start_time = survey_info['s0_start_time'].tolist()
	end_time = survey_info['s0_end_time'].tolist()
	subj_id = survey_info.index.values.tolist()
	survey_ip = survey_info['s0_person_ip'].tolist()
	mse_time = server_info['time']
	server_ip = server_info['person_ip'][0]

	start_time_s = []
	end_time_s = []
	subj_pos = -1
	ref_datetime = dt.datetime.strptime('1969-12-31 16:00', '%Y-%m-%d %H:%M')

	for i in start_time:
		y = dt.datetime.strptime(i, '%d/%m/%Y %H:%M')
		diff_time = y - ref_datetime
		start_time_s.append(int(diff_time.total_seconds()))

	for i in end_time:
		y = dt.datetime.strptime(i, '%d/%m/%Y %H:%M')
		diff_time = y - ref_datetime
		end_time_s.append(int(diff_time.total_seconds()))

	for i in pl.arange(0, len(start_time_s)):
		if start_time_s[i] <= int(mse_time) <= end_time_s[i]:
			if server_ip == survey_ip[i]:
				subj_pos = subj_id[i]

	return subj_pos


def reorder_data(data):
	""" This function reorder the data from mouse movements file according to the number of frames.
	Parameters
	----------
	data: dataframe
	  columns of file to order.
	Returns
	-------
	data: dataframe
	  reorder data.
	lost_samples: float
		percentage of lost samples
	"""

	data = data.sort_values(['t', 'x', 'y'])
	print(data)
	# print(data)
	# index = data[0]
	# lost_samples = sum(pl.diff(index) - 1)

	# return data, [float(lost_samples) / float(len(index))]
	return data

def get_parameters(data):
	""" This function gets the parameters of mouse movements from the dataframe
	Parameters
	----------
	data: dataframe
	  columns of file.
	Returns
	-------
	track_variables: dataframe
	  indexes: x (int px), y (int px), t (int s), events (int), items (str)
	"""

	item = data[len(data.columns) - 10]
	item = pl.array(item)

	x = data[len(data.columns) - 7]
	x = pl.array(x)
	x = x[~pl.isnan(x)]

	y = data[len(data.columns) - 6]
	y = pl.array(y)
	y = y[~pl.isnan(y)]

	t = data[len(data.columns) - 1]
	t = pl.array(t)
	t = (t - int(t[0])) / 1000.  # Absolute to relative time in s
	t = t[pl.find(~pl.isnan(x))]

	events = data[len(data.columns) - 11]  # events all mousemove=0 mousedown=1 mouseup=4
	events = pl.array(events)
	events = events[pl.find(~pl.isnan(x))]

	item = item[pl.find(~pl.isnan(x))]

	track_variables = pd.DataFrame({'x': [x],
									'y': [y],
									't': [t],
									'events': [events],
									'items': [item]
									})
	return track_variables


def correct_parameters(track_variables):
	""" This function correct x, y, t.
	Remove sequential frames with the same position (x,y) and with the same time.
	In this case we lose information.
	Parameters
	----------
	track_variables: dataframe
	  use indexes x, y, t
	Returns
	-------
	track_variables: dataframe
	  correct the indexes x, y, t
	samples_correction: int
	  Number of samples cut.
	"""

	x = track_variables['x'][0].tolist()
	y = track_variables['y'][0].tolist()
	t = track_variables['t'][0].tolist()
	total_var = [[x[i], y[i], t[i]] for i in pl.arange(0, len(t))]
	total_var = pl.array(total_var)
	t_error = pl.find(pl.diff(total_var[:, 2]) == 0)
	total_var = pl.delete(total_var, t_error, 0)
	x_error = pl.find(pl.diff(total_var[:, 0]) == 0)
	y_error = pl.find(pl.diff(total_var[:, 1]) == 0)
	xy_error = set(x_error).intersection(y_error)
	xy_error = list(xy_error)
	samples_correction = len(t_error) + len(xy_error)
	total_var = pl.delete(total_var, xy_error, 0)
	x = total_var[:, 0]
	y = total_var[:, 1]
	t = total_var[:, 2]
	for i in pl.find(pl.diff(x) == 0):
		if i in pl.find(pl.diff(y) == 0):
			print("Nr samples with repeated position: ", i)
	if len(pl.find(pl.diff(t) == 0)) != 0:
		print("Nr samples with repeated time: ", len(pl.find(pl.diff(t) == 0)))

	track_variables['x'] = [x]
	track_variables['y'] = [y]
	track_variables['t'] = [t]

	return track_variables, samples_correction


def extract_item_number(track_variables):
	""" This function gets the question number and its answer.
	Parameters
	----------
	track_variables: dataframe
	  string with mouse item. (eg answer465687X11X85SQ64-A2)
	Returns
	-------
	track_variables: dataframe
	  change index items (int question number (eg 64))
	  add index answers (int question answer (eg 2))
	  add t_items (int time where item begins)
	"""

	t = track_variables['t'][0].tolist()
	items = track_variables['items'][0].tolist()
	ret_items = pl.zeros(len(items))
	ret_answer = pl.zeros(len(items))
	t_items = pl.zeros(len(items))
	for i, ix in enumerate(items):
		if 'SQ' in ix:
			# Lime survey notation uses strings such as:
			# answer465687X11X85SQ64-A2
			# javatbd465687X11X85SQ64
			# The relevant items comes after SQ

			ret_items[i] = int(ix.split('-')[0].split('SQ')[1])
			t_items[i] = t[i]
			if len(ix.split('-')) == 2:
				ret_answer[i] = int(ix.split('-')[1][1])
			else:
				ret_answer[i] = -1

		else:
			ret_items[i] = -1
			t_items[i] = t[i]
			ret_answer[i] = -1

	track_variables['items'] = [ret_items.astype('int')]
	track_variables['answers'] = [ret_answer.astype('int')]
	track_variables['t_items'] = [t_items]
	return track_variables


def get_new_item_ix(track_variables):
	""" This function simplify the items analysis giving the items order and removing the items 0.
	Parameters
	----------
	track_variables: dataframe
	  to use index items
	Returns
	-------
	i: array
	  int with index of item change.
	items_order: array
	  int with items ordered.
	"""

	items = track_variables['items'][0].tolist()
	items = pl.array(items)
	i = pl.find(pl.diff(items) != 0) + 1
	items_order = items[i].tolist()

	if len(i) > 0:
		items_order.insert(0, items[i[0] - 1])

	return i, items_order


def count_items(items_order, context_variables, survey):
	""" This function count the items in each group and recognize the group.
	This step is important to check if the group is complete.
	Parameters
	----------
	items_order: array
	  int with ordered items.
	context_variables: dataframe
	  to add nr_items and group
	survey: int
	  to identify questions of groups
	Returns
	-------
	context_variables: dataframe
	  indexes: nr_items (int number of items in group), group (string group name)
	"""

	item0 = 0
	group = None
	items_order = list(items_order)
	my_dict = {i: items_order.count(i) for i in items_order if i != -1}  # item: nr times
	nr_items = len(my_dict)

	for j in my_dict:
		item0 = j

	if survey == 465687:
		if 0 < item0 < 19:
			group = "OldMaximizer"
		elif 18 < item0 < 31:
			group = "NEO"
		elif 30 < item0 < 65:
			group = "NewMaximizer"
		elif nr_items == 0:
			group = "no items"
		else:
			group = "no group recognized"
	if survey == 935959:
		if 0 < item0 < 19:
			group = "OldMaximizer"
		elif 18 < item0 < 79:
			group = "NEO"
		elif 78 < item0 < 99:
			group = "ARES"
		elif 98 < item0 < 115:
			group = "SPF"
		elif 114 < item0 < 142:
			group = "HAKEMP"
		elif 141 < item0 < 176:
			group = "NewMaximizer"
		elif nr_items == 0:
			group = "no items"
		else:
			group = "no group recognized"

	context_variables['nr_items'] = [nr_items]
	context_variables['group'] = [group]

	return context_variables