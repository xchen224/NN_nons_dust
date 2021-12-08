import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy import stats

def density(ax, x_data, y_data, **kwargs):

	''' 
	Function to create bi-dimensional histograms
	
	:param ax: figure instance to create a plot
	:type ax: figure-object
	:param x_data: array containing data associated with X axis
	:type x_data: array-like
	:param y_data: array containing data associated with Y axis
	:type y_data: array-like
	:param x_label: label for X axis
	:type x_label: string, optional
	:param y_label: label for Y axis
	:type y_label: string, optional
	:param color_bar: name of color bar (https://matplotlib.org/examples/color/colormaps_reference.html)
	:type color_bar: matplotlib-color, optional
	:param plot_title: title of the plot
	:type plot_title: string, optional
	:returns: ax, figure instance containing the scatter density plot
	:rtype: matplotlib.figure.Figure object
	'''
	
	color_bar 	= kwargs.get('color_bar', plt.cm.jet)
	
	
	#-- get a mask with those elements posible to compare (non-nan values)
	mask = ~np.isnan(x_data) & ~np.isnan(y_data)
	
	x_data_aux = x_data[mask]
	y_data_aux = y_data[mask]
	n_colocations = len(mask[mask==True])
	
	
	# data range
	minx = math.floor(x_data_aux.min())
	maxx = math.ceil(x_data_aux.max())
	miny = math.floor(y_data_aux.min())
	maxy = math.ceil(y_data_aux.max())
	
	xyrange = [[minx,maxx],[miny,maxy]] 
	#xyrange = [[0,4],[0,4]] 
	
	data_source = np.zeros((n_colocations, 2))
	data_source[:,0] = x_data
	data_source[:,1] = y_data
	
	# determine number of bins based on the Freedman Diaconis rule
	IQRs = stats.iqr(data_source, axis=0)
	hs = 2 * IQRs* pow(n_colocations, -0.3333333333333333)
	bw = int(max((x_data_aux.max() - x_data_aux.min())/hs[0], 
		     (y_data_aux.max() - y_data_aux.min())/hs[1]))
	
	#bw = 10
	bins = [bw,bw]
	
	# define density threshold 
	thresh = 2
	
	# histogram of the data
	hh, locx, locy = scipy.histogram2d(x_data_aux, y_data_aux, 
					   range=xyrange, bins=bins, normed=True)
	
	# fill the areas with low density by NaNs
	hh[hh < thresh] = np.nan 
	
	his = ax.imshow(np.flipud(hh.T), cmap=color_bar, extent=np.array(xyrange).flatten(), 
					interpolation='nearest', zorder=2, alpha=.8)

	cb1 = plt.colorbar(his,ax=ax)
	cb1.set_label('Density')

	
	
	return ax



def scatter(ax, in_x_data, in_y_data, label_p = 'upper left', pos0=None, fig=None, fsize=10, outputerr=False, \
        regress_line=True, one2one_line=True, color='b', mre=True, marker=None, cbar_ticks=None, \
        cbar_label=None,cticklabel=None, label=None, thresh=1.e-6, **kwargs):
    ''' 
    Function to create scatter plots
    '''
 
    # copy and flatten data
    x_data = in_x_data.flatten()
    y_data = in_y_data.flatten()

    #-- get a mask with those elements posible to compare (non-nan values)
    mask = np.logical_and(np.logical_not(np.isnan(x_data)), np.logical_not(np.isnan(y_data)))
    n_colocations = len(mask[mask==True])
    x_data = x_data[mask]
    y_data = y_data[mask]
    
    #-- liner regression
    slope, intercept, correlation, p_value_slope, std_error = stats.linregress(x_data, y_data)

    #-- Calculates a Pearson correlation coefficient and the p-value for testing non-correlation
    r, p_value = stats.pearsonr(x_data, y_data)

    rmse   = np.sqrt( np.mean( np.power((y_data-x_data),2) ) )
    mean_x = np.mean(x_data)
    mean_y = np.mean(y_data)
    std_x  = np.std(x_data, ddof=1)
    std_y  = np.std(y_data, ddof=1)
    
    #-- create scatter plot
    if marker is None:
       if label is not None:
          paths = ax.scatter(x_data, y_data, s=5, alpha=0.7, c=color,label=label, **kwargs)
       else:  
          if isinstance( color, str ): 
             paths = ax.scatter(x_data, y_data, s=5, alpha=0.7, c=color, **kwargs)
          else:
             print('cbar_tick',cbar_ticks)
             if len(cbar_ticks) > 12:
                cmap = plt.cm.get_cmap('tab20', len(cbar_ticks))
             else:
                cmap = plt.cm.get_cmap('Paired', len(cbar_ticks))
             paths = ax.scatter(x_data, y_data, s=5, c=color,  alpha=0.5, cmap=cmap,
                                vmin=cbar_ticks[0], vmax=cbar_ticks[-1], **kwargs)
             cb = fig.colorbar(paths, ax=ax, shrink=0.98, pad=0.03, ticks=cbar_ticks)                            
             cb.set_label(cbar_label)     
             if cticklabel is not None:
                cb.ax.set_yticklabels(cticklabel)
    else:
       if label is not None:
          paths = ax.scatter(x_data, y_data, s=10, alpha=0.5, marker=marker, edgecolors=color, c='', linewidths=0.5, label=label, **kwargs)
       else:
          paths = ax.scatter(x_data, y_data, s=10, alpha=0.5, marker=marker, edgecolors=color, c='', linewidths=0.5, **kwargs) 
           
    
    #-- add slope line
    min_x = np.nanmin(x_data)
    max_x = np.nanmax(x_data)
    min_y = np.nanmin(y_data)
    max_y = np.nanmax(y_data)
    #x = np.array((min_x-np.absolute(min_x),max_x+np.absolute(max_x)))
    min_all = min(min_x, min_y)
    max_all = max(max_x, max_y)
    x = np.linspace(min_all, max_all, num=1000, endpoint=True)
    y = (slope * x) + intercept
    if regress_line:
        if isinstance( color, str ):
           ax.plot(x, y, '-', color=color, linewidth=1.0)
        else:
           ax.plot(x, y, '-', color='r', linewidth=1.0)   
    if one2one_line:
        ax.plot(x, x, '--', color='black', linewidth=0.8)
    
    #-- create strings for equations in the plot
    correlation_string = "R = {:.2f}".format(r)
    
    sign = " + "
    if intercept < 0:
        sign = " - "
        
    #lineal_eq = "y = " + str(round(slope, 2)) + "x" + sign + str(round(abs(intercept), 2))
    lineal_eq = "y = " + str(round(slope, 2)) + "x" + sign + "{:.2e}".format(abs(intercept))
    rmse_coef = "RMSE = {:.2e}".format(rmse)

    if p_value >= 0.05:
        p_value_s = "(p > 0.05)"
    else:
        if p_value < 0.01:
            p_value_s = "(p < 0.01)"
        else:
            p_value_s = "(p < 0.05)"

    n_collocations = "N = " + str(n_colocations)
    x_mean_std = "x: {:.2e}".format(mean_x) + " $\pm$ " + "{:.2e}".format(abs(std_x))
    y_mean_std = "y: {:.2e}".format(mean_y) + " $\pm$ " + "{:.2e}".format(abs(std_y))

    # mean relative error
    if mre: 
       #idx0 = np.nonzero(x_data)
       if thresh is not None:
          idx0 = np.where(np.abs(x_data) > thresh)[0]
          mre_value = np.nanmean(np.abs(y_data[idx0] - x_data[idx0]) / np.abs(x_data[idx0])) * 100 
       else:
          mre_value = np.nanmean(np.abs(y_data - x_data) / np.abs(x_data)) * 100
       mre_str = "MRE = {:.2f}%".format(mre_value)
    else:
       idx0 = np.nonzero(x_data) 
       mre_value = np.nanmean(y_data[idx0] - x_data[idx0]) / np.nanmean(x_data[idx0]) * 100 
       mre_str = "RME = {:.2f}%".format(mre_value)
          
    
    print(correlation_string + ' ' + p_value_s)
    fsize = fsize

    if (label_p == 'upper left'):
        equations0 = \
                correlation_string + ' ' + p_value_s + '\n' + \
                x_mean_std + '\n' + \
                y_mean_std + '\n' + \
                lineal_eq  + '\n' + \
                rmse_coef  + '\n' + \
                mre_str    + '\n' + \
                n_collocations
                
        if pos0 is None:
        # upper left
          posXY0      = (0, 1)
          posXY_text0 = (5, -5)
          if isinstance( color, str ):
             ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', \
                  xycoords='axes fraction', size=fsize, color=color, textcoords='offset points')
          else:
             ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', \
                  xycoords='axes fraction', size=fsize, color='r', textcoords='offset points')        
        else:
          posXY0      = (0, 1+pos0)
          posXY_text0 = (5, -5)
          if isinstance( color, str ):
             ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', \
                  xycoords='axes fraction', size=fsize, color=color, textcoords='offset points')
          else:
             ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', \
                  xycoords='axes fraction', size=fsize, color='r', textcoords='offset points')        
                  
    elif (label_p == 'lower right'):
        equations0 = n_collocations + '\n' + \
                mre_str    + '\n' + \
                rmse_coef  + '\n' + \
                x_mean_std + '\n' + \
                y_mean_std + '\n' + \
                lineal_eq  + '\n' + \
                correlation_string + ' ' + p_value_s

        # lower right
        if pos0 is None:
           posXY0      = (1, 0)
           posXY_text0 = (-5, 5)
           if isinstance( color, str ):
              ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='right', \
                xycoords='axes fraction', size=fsize, color=color, textcoords='offset points')
           else:
              ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='right', \
                xycoords='axes fraction', size=fsize, color='r', textcoords='offset points')     
        else:
           posXY0      = (1, 0+pos0)
           posXY_text0 = (-5, 5)
           if isinstance( color, str ):
              ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='right', \
                xycoords='axes fraction', size=fsize, color=color, textcoords='offset points') 
           else:
              ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='right', \
                xycoords='axes fraction', size=fsize, color='r', textcoords='offset points')            

    elif (label_p == 'positive'):
        equations1 = correlation_string + ' ' + p_value_s + '\n'+ \
                lineal_eq  + '\n' + \
                rmse_coef  + '\n'
        equations2 = n_collocations + '\n' + \
                x_mean_std + '\n' + \
                y_mean_std
            
        posXY1      = (0, 1)
        posXY_text1 = (5, -5)
        ax.annotate(equations1, xy=posXY1, xytext=posXY_text1, va='top', \
                xycoords='axes fraction', textcoords='offset points')
    
        posXY2      = (1, 0)
        posXY_text2 = (-5, 5)
        ax.annotate(equations2, xy=posXY2, xytext=posXY_text2, va='bottom', ha='right', \
                xycoords='axes fraction', textcoords='offset points')
    else:
        
        '!!! scatter: label_p error !!!'
        exit()
    
    if outputerr:
       return paths, slope, intercept, r, rmse, mre_value
    else:
       return paths, slope, intercept   
