import numpy as np
import scipy as sp
import pylab as pl


def get_path_smooth_s(t, x, y, begin_interp=0, end_interp=-1, stol=0.1):
	""" This function computes a spatial interpolation.
	Parameters
	----------
	t: array
	  float timestamp (seconds
	x: array
	  int mouse cursor x position.
	y: array
	  int mouse cursor y position.
	begin_interp: int
	  index in array to begin interpolation (if we want just a part).
	end_interp: int
	  index in array to end interpolation (if we want just a part).
	stol: float
	  tolerance (samples/pixel)
	Returns
	-------
	ss: array
	  float distance values in interpolation.
	xs: array
	  float mouse cursor x position spatial interpolated.
	ys: array
	  float mouse cursor y position spatial interpolated.
	vs: array
	  float spatial velocity
	angle_value: array
	  float angular results
	curvature: array
	  float curvature results
	"""

	t = t[begin_interp:end_interp]
	x = x[begin_interp:end_interp]
	y = y[begin_interp:end_interp]

	s = get_s(x, y)
	# print('x')
	# print(pl.diff(x))
	# print('y')
	# print(pl.diff(y))
	# print(s)
	# print(min(pl.diff(s)))
	ss = pl.arange(s[0], s[-1], stol)

	if (len(x) > 4 ):
		splxs = sp.interpolate.UnivariateSpline(s, x, k=2, s=2)
		xs = splxs(ss)
		splys = sp.interpolate.UnivariateSpline(s, y, k=2, s=2)
		ys = splys(ss)
		splts = sp.interpolate.UnivariateSpline(s, t, k=1, s=0)
		ts = splts(ss)
	else:
		xs = []
		ys = []
		ts = []

	return xs, ys, ts


def get_path_smooth_t(t, x, y, begin_interp=0, end_interp=-1, ttol=0.1):
		""" This function computes a temporal interpolation.
		Parameters
		----------
		t: array
		  float timestamp (seconds)
		x: array
		  int mouse cursor x position.
		y: array
		  int mouse cursor y position.
		begin_interp: int
		  index in array to begin interpolation (if we want just a part).
		end_interp: int
		  index in array to end interpolation (if we want just a part).
		ttol: float
		  tolerance (samples/second)
		Returns
		-------
		tt: array
		  float time values in interpolation.
		xt: array
		  float mouse cursor x position spatial interpolated.
		yt: array
		  float mouse cursor y position spatial interpolated.
		"""

		t = t[begin_interp:end_interp]
		x = x[begin_interp:end_interp]
		y = y[begin_interp:end_interp]

		t = t[:len(x)]
		_tt = pl.arange(t[0], t[-1], ttol)

		if len(x) > 4:
			splxt = sp.interpolate.UnivariateSpline(t, x, k=1, s=5)
			xt = splxt(_tt)
			splyt = sp.interpolate.UnivariateSpline(t, y, k=1, s=5)
			yt = splyt(_tt)
		else:
			_tt = []
			xt = []
			yt = []

		return _tt, xt, yt


def get_s(x, y):
	""" This function calculates the distance traveled.
	Parameters
	----------
	x: array
	  int mouse cursor x position.
	y: array
	  int mouse cursor y position.
	Returns
	-------
	s: array
	  float cumulative distance traveled.
	"""
	ds = pl.sqrt(pl.diff(x)**2+pl.diff(y)**2)
	# print(pl.unique(ds))
	# ds=pl.unique(ds)
	s = pl.cumsum(pl.concatenate(([0], ds)))

	# print(min(pl.diff(s)))


	return s


def get_v(t, s):
	""" This function calculates the velocity in time.
	Parameters
	----------
	t: array
	  float timestamp (seconds)
	s: array
	  float cumulative sum of distance traveled.
	Returns
	-------
	diff(s)/diff(t[:len(s)]): array
	  velocity (pixeis/sec)
	"""

	return pl.diff(s)/pl.diff(t[:len(s)])


def generate_circle(r=1.0, s=0.01):
	""" This function generate a circle to test functions.
	Parameters
	----------
	r: array
	  radius of circle
	s: float
	  step for time.
	Returns
	-------
	t: array
	  time
	x: array
	  values of x position
	y: array
	  values of y position
	"""

	t = pl.arange(0, 2*pl.pi, s)
	x = pl.sin(t)*r
	y = pl.cos(t)*r

	return t, x, y


def generate_random_circle():
	""" This function generate a random circle (r=1) to test functions.
	Returns
	-------
	t: array
	  time
	x: array
	  values of x position
	y: array
	  values of y position
	"""

	exp_range = np.random.exponential(scale=0.1, size=100)
	d = pl.cumsum(exp_range[exp_range > 0.05])
	t = d[d < 2*pl.pi]

	x = pl.sin(t)
	y = pl.cos(t)

	return t, x, y


def check_incremental_s(s):
	""" This function verifies if s is gradually increasing.
	Parameters
	----------
	s: array
	  values to analyse.
	Returns
	-------
	all(diff(s) > 0.0): bool
	  True if array is gradually increasing.
	"""

	return all(pl.diff(s) > 0.0)