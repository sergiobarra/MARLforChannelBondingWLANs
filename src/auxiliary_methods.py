# Plot pdf of a output data
def plot_data_pdf(data, xlabel, ylabel, xlim=False, xlog=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print('Plotting output pdf...')
    sns.set_style("white")
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    plt.figure(figsize=(4,3), dpi= 80)
    #sns.distplot(data,  bins=20, color="dodgerblue", label="Compact", **kwargs)
    ax = sns.distplot(data, hist=False, rug=True);
    if xlim:
        plt.xlim(xlim)
       
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    plt.grid()
    
    if xlog:
        print('- converting x-axis to log scale...')
        ax.set_xscale('log')
    
# Compute model error
def compute_model_error(model, x, y):
    import numpy as np
    from sklearn import metrics
    y_pred = model.predict(x)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    # print('- RMSE: %.4f' % rmse)
    # print('- MAE: %.4f' % mae)
    return rmse, mae

# Plot NN error evolution
def plot_nn_error(history):
    import matplotlib.pyplot as plt
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()