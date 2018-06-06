import numpy as np

def oneHot(y, cols=None):
    if (cols < 1):
        cols = np.max(y)
    n = y.shape[0]
    z = np.zeros((n, cols))
    z[np.arange(n), (y-1)] = 1
    return z

def customPredict(scores, y, y2=None, frames=10):
    y_f = y[::frames]
    y2_f = y2[::frames]
    
    predictions = np.argmax(scores, axis=1)
    acc = np.sum(predictions == y-1) / len(y)
    
    # Averaging
    N, T = scores.shape
    scores_reshaped = scores.reshape((N//10, 10, T))
    avg_scores = np.mean(scores_reshaped, axis=1)
    
    # Primary accuracy
    avg_predictions = np.argmax(avg_scores, axis=1)
    matches = avg_predictions == y_f-1  # y is 1-indexed
    acc1 = np.sum(matches)/len(matches)
    
    # Dual accuracy
    acc2 = None
    if not y2 is None:
        matches2 = avg_predictions == y2_f-1
        acc2 = np.sum(matches + matches2)/len(matches)  # + acts as an or operator
        
    # Other metrics
    
    confusion_mat = np.zeros((18,18))
    for i in range(len(avg_predictions)):
        # Rows are predictions, cols are actual
        confusion_mat[avg_predictions[i]][y_f[i]-1] += 1
    true_pos = np.diag(confusion_mat)
    false_pos = np.sum(confusion_mat, axis=0) - true_pos
    false_neg = np.sum(confusion_mat, axis=1) - true_pos
    avg_precision = np.nanmean(true_pos / (true_pos + false_pos))
    avg_recall    = np.nanmean(true_pos / (true_pos + false_neg))

    
    metrics = {}
    ###
    metrics['acc'] = acc
    metrics['avg_acc'] = acc1
    metrics['avg_acc_2'] = acc2
    
    metrics['avg_scores'] = avg_scores
    metrics['predictions'] = predictions
    metrics['avg_predictions'] = avg_predictions
    
    metrics['confusion_mat'] = confusion_mat
    metrics['avg_precision'] = avg_precision
    metrics['avg_recall'] = avg_recall
    ###
    return metrics