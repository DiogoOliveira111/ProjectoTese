from WBMTools.sandbox.data_processing import reorder_data, correct_parameters
from WBMTools.sandbox.pathmath import get_s, get_v, get_path_smooth_t, get_path_smooth_s
from WBMTools.sandbox.auxiliar_function import round_dig
import pylab as pl
import pandas as pd
from scipy import signal
from novainstrumentation import smooth
from scipy.signal import decimate



def interpolate_data(track_variables, t_abandon, t_crop=pl.Inf, begin=0, end=-1):
	""" This function compute the spatial and temporal data.
	Parameters
	----------
	track_variables: dataframe
	  to use indexes time, x and y
	context_variables: dataframe
	  to join features
	t_abandon: float
	  time established to consider an abandon event
	t_crop: int
	  time to cut data
	begin: int
	  begin of data analysis
	end: int
	  end of data analysis
	Returns
	-------
	time_variables: dataframe
	  indexes
	  xt: array (x interpolated in time (px))
	  yt: array (y interpolated in time (px))
	  tt: array (time interpolated (sec))
	  vt: array (velocity in time (px/sec))
	  vx: array (horizontal velocity in time (px/sec))
	  vy: array (vertical velocity in time (px/sec))
	  a: array (acceleration (px/sec²))
	  jerk: array (jerk (px/sec³))
	space_variables: dataframe
	  indexes
	  xs: array (x interpolated in space (px))
	  ys: array (y interpolated in space (px))
	  l_strokes: array (distance for stroke (px/items))
	  straightness: array (real distance/shorter distance (px/px))
	  jitter: array (tremors analysis, relation between original and smooth path)
	  s: array (cumulative distance with mouse (px))
	  angles: array (angle spacial movement (rad))
	  w: array (angular velocity (rad/sec))
	  curvatures: array (1/R curvature spatial movement (rad/px))
	  var_curvatures: array (curvature variation in space (rad/px²))
	context_variables: dataframe
	  add index nr_pauses: int total number of pauses
	  add index t_pauses: array time of each pause
	"""
	#reorder data:
	t_v = reorder_data(track_variables)
	t = t_v['t'].tolist()[begin:end]
	x = t_v['x'].tolist()[begin:end]
	y = t_v['y'].tolist()[begin:end]



	#time variables
	x = pl.array(x)
	y = pl.array(y)
	_s = get_s(x, y)
	x_f = smooth(x)
	y_f = smooth(y)
	s_f = get_s(x_f, y_f)
	jitter = s_f[-1] / _s[-1] #é suposto ser um valor unico?

	t_temp = t

	# Detect time interpolation factor by mean of dt
	dt = pl.diff(t)
	dt = dt[dt != 0]
	min_t = min(dt)
	dig_int = round_dig(min_t)
	interp_f = round(min_t, dig_int)
	if t[-1] / interp_f > 200000000:  # 200000000 memory error
		interp_f = t[-1] / 200000000

	# Detect spatial interpolation factor by min of ds
	_s = get_s(x, y)
	ds = pl.diff(_s)
	ds = ds[ds != 0]
	min_s = min(ds)
	dig_int_s = round_dig(min_s)
	interp_s = round(min_s, dig_int_s)
	# print(_s[-1] / interp_s)
	if _s[-1] / interp_s > 200000000:  # 200000000 memory error - foi posto agora porque estava a dar mal para vectores grandes
		interp_s = _s[-1] / 200000000
	# print(interp_s)

	# find pauses
	# i_inter = list(find(diff(t) <= 1))
	i_inter = pl.arange(0, len(t))
	t_pauses, t_all, xt, yt, xs, ys, ts, angles, w, curvatures, var_curvatures, l_strokes, straightness = ([] for _ in
																										   range(13))
	inter = []
	t_inter_total = []
	i_inter_total = []
	for i in pl.arange(0, len(i_inter)-1):
		if t[i_inter[i+1]]-t[i_inter[i]] < 1:
			inter.append(i_inter[i+1])
			if i == len(i_inter)-2:
				inter.insert(0, inter[0] - 1)
				t_inter_total.append(pl.array(t)[inter])
				i_inter_total.append(inter)
		elif len(inter) > 0:
			inter.insert(0, inter[0] - 1)
			t_inter_total.append(pl.array(t)[inter])
			i_inter_total.append(inter)
			inter = []

	for i in pl.arange(0, len(t_inter_total)-1):
		if len(t_pauses) == 0:
			t_pauses.append(t_inter_total[i][0])
		else:
			t_pauses.append(t_inter_total[i+1][0] - t_inter_total[i][-1])
	nr_pauses = len(t_pauses)
	# interpolate/stroke
	t_interac_acum = 0
	for i in pl.arange(0, len(t_inter_total)):
		if (i_inter_total[i][-1] - i_inter_total[i][0]) > 2:
			begin_t = i_inter_total[i][0]
			end_t = i_inter_total[i][-1]
			t_interac_acum += t[end_t]-t[begin_t]
			if t_interac_acum < t_crop:
				t_slice, xt_slice, yt_slice = get_path_smooth_t(t, x, y, begin_t, end_t+1, ttol=interp_f)
				t_all = pl.concatenate((t_all, t_slice))
				xt = pl.concatenate((xt, xt_slice))
				yt = pl.concatenate((yt, yt_slice))
				xs_slice, ys_slice, ts_slice = \
					get_path_smooth_s(t_temp, x, y, begin_t, end_t+1, stol=interp_s)
				xs = pl.concatenate((xs, xs_slice))
				ys = pl.concatenate((ys, ys_slice))
				ts = pl.concatenate((ts, ts_slice))
				s_strokes = get_s(xs_slice, ys_slice)
				if s_strokes[-1] > 0:
					l_strokes.append(s_strokes[-1])
					straightness.append((pl.sqrt(((ys_slice[-1] - ys_slice[0]) ** 2) + ((xs_slice[-1] - xs_slice[0])
																						** 2))) / s_strokes[-1])

	ss = get_s(xs, ys)
	#print(ss)

	# angle = atan(dy/dx)
	# unwrap removes discontinuities.
	angle_value = smooth(pl.unwrap(pl.arctan2(pl.diff(ys) / pl.diff(ss), pl.diff(xs) / pl.diff(ss))))

	# angular velocity
	w = angle_value / (pl.diff(ss) / pl.diff(ts))

	# c = (dx * ddy - dy * ddx) / ((dx^2+dy^2)^(3/2))
	curvature_top = (pl.diff(xs) / pl.diff(ss))[:-1] * pl.diff(pl.diff(ys) / pl.diff(ss)) / (pl.diff(ss)[:-1]) - \
					(pl.diff(ys) / pl.diff(ss))[:-1] * pl.diff(pl.diff(xs) / pl.diff(ss)) / (pl.diff(ss)[:-1])
	curvature_bottom = ((pl.diff(xs) / pl.diff(ss)) ** 2 + (pl.diff(ys) / pl.diff(ss)) ** 2) ** (3 / 2.0)
	curvature = pl.array(curvature_top / curvature_bottom[:-1])
	var_curvature = pl.array(pl.diff(curvature) / pl.diff(ss)[:-2])

	st = get_s(xt, yt)
	#print(st)
	# Save t_pauses > 1 sec
	# Save t_pauses < abandon
	t_pauses = pl.array(t_pauses)
	t_pauses = t_pauses[t_pauses > 1.]
	t_pauses = t_pauses[t_pauses < t_abandon]

	# velocity moving and total
	vt_moving = get_v(t_all, st)

	t = pl.arange(t_temp[0], t_temp[-1], interp_f)
	vt_moving = list(vt_moving)
	vx = list(abs(pl.diff(xt) / pl.diff(t_all)))
	vy = list(abs(pl.diff(yt) / pl.diff(t_all)))

	for i in pl.arange(0, len(i_inter_total)-1):
		if i == 0:
			begin = [0]
			end = pl.find(t - t_temp[i_inter_total[i][0]] > interp_f)
		else:
			begin = pl.find(t - t_temp[i_inter_total[i-1][-1]] > interp_f)
			end = pl.find(t - t_temp[i_inter_total[i][0]] > interp_f)
		zero_v = pl.arange(begin[0], end[0])
		for j in zero_v:
			vt_moving.insert(j, 0)
			vx.insert(j, 0)
			vy.insert(j, 0)

	vt_moving.extend(pl.zeros(len(t) - len(vt_moving)))
	vx.extend(pl.zeros(len(t) - len(vx)))
	vy.extend(pl.zeros(len(t) - len(vy)))
	vt = pl.array(vt_moving)
	vx = pl.array(vx)
	vy = pl.array(vy)

	a = pl.diff(vt) / pl.diff(t)
	jerk = pl.diff(a) / pl.diff(t[:-1])

	space_variables = dict(			xs= xs, #tirei decimaçao de 20 "xs[::20]
									ys= ys,
									l_strokes=pl.array(l_strokes),
									straightness= pl.array(straightness),
									jitter= jitter,
									s= st,
									ss= ss,
									angles= angle_value,
									w= w,
									curvatures= curvature,
									var_curvatures=var_curvature
								)

	time_variables = dict(
						  xt= xt,
	                      yt= yt,
	                      tt= t_all,
	                      ttv= t,
	                      vx=vx,
					      vy=vy,
	                      #vx= decimate(vx,20, ftype='fir'),
	                      #vy= decimate(vy, 20, ftype='fir'),
	                      vt= vt,
	                      a= a,
	                      jerk=jerk
	)

	# context_variables['nr_pauses'] = {nr_pauses /
	# 								  float(context_variables['nr_items'][0])}
	# context_variables['t_pauses'] = [t_pauses]
	# context_variables['t_interaction'] = [t_interac_acum]

	return time_variables, space_variables