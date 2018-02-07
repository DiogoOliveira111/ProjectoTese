import pylab as pl
import os
import numpy as np


def files_with_pattern(directory):
	for path, dirs, files in os.walk(directory):
		for i in files:
			yield os.path.join(path, i)


def get_files(directory, l_strings=''):
	""" This function receives one string or a list of strings.
	Parameters
	----------
	directory: string
	  the file directory.
	l_strings: string
	  code in file.
	Returns
	-------
	list_files: array-like
	  the files
	"""
	if pl.is_string_like(l_strings):
		return [i for i in files_with_pattern(directory) if l_strings in i]
	else:
		list_files = [i for i in files_with_pattern(directory)]
		for include_string in l_strings[:]:
			list_files = [i for i in list_files if include_string in i]
	return list_files


def crop_data(track_variables, time):
	""" This function just keep the data that we want to analyse (in order to study the interaction during x time).
	Parameters
	----------
	track_variables: dataframe
	  crop all tracking variables
	time: int
	  desired time of interaction
	Returns
	-------
	track_variables: dataframe
	  new set of track_variables
	"""

	limit = pl.find((pl.cumsum(pl.diff(track_variables['t'][0])[1:])) <= time)[-1]
	for i in track_variables.columns:
		track_variables[i][0] = track_variables[i][0][:limit+1]

	return track_variables


def get_statistics(variable, zero=0):
	""" This function analyse statistically variables.
	Parameters
	----------
	variable: array
	  variable to analyse.
	zero: int
	  consider values equal to 0.
	Returns
	-------
	var[argmax(var)]: float
	  maximum value of array
	min_value: float
	  minimum value of array
	mean(var): float
	  mean value of array
	std(var): float
	  standard deviation of array
	"""
	variable = pl.array(variable)
	if zero == 0:
		min_value = variable[pl.argmin(variable)]
	else:
		min_value = variable[variable > 0][pl.argmin(pl.array(variable[variable > 0]))]

	return variable[pl.argmax(variable)], min_value, pl.mean(variable), pl.std(variable)


def round_dig(x):
	""" This function returns the number of significant numbers. To round values.
	Parameters
	----------
	x: float
	  value to analyse
	Returns
	-------
	int significant digits
	"""
	return -int(pl.floor(pl.log10(abs(x))))


def reject_outliers(data, m=1):
	""" This function reject outliers based on mean and std
	Parameters
	----------
	data: list
	  list to reject outliers
	m: int
	  factor to remove the outliers
	Returns
	-------
	data: array
	  list without outliers
	"""
	data = pl.array(data)
	return data[abs(data - pl.mean(data)) < m * pl.std(data)]


def plot_confusion_matrix(cm, classes, frame, normalize=False, cmap=pl.plt.cm.PRGn):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	pl.plt.imshow(cm, cmap=cmap, interpolation='none')

	pl.plt.colorbar()
	tick_marks = np.arange(0.5, len(classes) + 0.5)
	tick_marks2 = np.arange(0.5, len(frame) + 0.5)
	pl.plt.xticks(tick_marks, classes, rotation=90, fontsize=5)
	pl.plt.yticks(tick_marks2, frame, fontsize=5)

	if normalize:
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	pl.plt.ylabel('n features label')
	pl.plt.xlabel('maximizer label')


def normalize(function):
	""" This function normalize an array based on his mean and std
			Parameters
			----------
			function: list
			  list to normalize
			Returns
			-------
			data: array
			  list normalized
			"""
	if np.mean(function) == 0:
		return np.zeros(len(function))
	else:
		return (function[:]-np.mean(function))/np.std(function)


def normalize_ms(function, m, s):
	""" This function normalize an array based on an introduced mean and std.
	   Parameters
			----------
			function: list
			  list to normalize
			m: float
			  mean
			s: float
			  std
			Returns
			-------
			data: array
			  list normalized
			"""
	if s == 0:
		return m
	else:
		return (function - m) / s