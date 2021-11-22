import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class CPlot:
    """Classification plot class with static methods"""
    
    @staticmethod
    def show_init_data_plot(X, y, cmap="tab10"):

        plt.title("Initial Data")
        scatter = plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap)
        plt.grid(True)
        plt.xlabel("X1")
        plt.ylabel("X2")
        # FIXME: version 0.20 
        # plt.legend(*scatter.legend_elements(), title="Class:")
        plt.show()

    @staticmethod
    def show_train_test_plots(model, X_train, y_train, X_test, y_test, 
                              title=None, cmap="tab10", proba=False, 
                              show_colorbar=True):

        step = 0.01

        x1_min = np.min([X_train[:,0].min(), X_test[:,0].min()])
        x1_max = np.max([X_train[:,0].max(), X_test[:,0].max()])

        x2_min = np.min([X_train[:,1].min(), X_test[:,1].min()])
        x2_max = np.max([X_train[:,1].max(), X_test[:,1].max()])

        x1_min = x1_min - (0.1*np.abs(x1_min))
        x1_max = x1_max + (0.1*np.abs(x1_max))
        x2_min = x2_min - (0.1*np.abs(x2_min))
        x2_max = x2_max + (0.1*np.abs(x2_max))

        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, step), 
                             np.arange(x2_min, x2_max, step))
        points = np.c_[xx.ravel(), yy.ravel()]

        if proba is True and hasattr(model, "predict_proba") and len(model.classes_) == 2:
            cmap = cm.bwr
            Z = model.predict_proba(points)[:, 1]
        elif proba is True and hasattr(model, "decision_function") and len(model.classes_) == 2:
            cmap = cm.bwr
            Z = model.decision_function(points)
        else:
            Z = model.predict(points)
        
        Z = Z.reshape(xx.shape)

        plt.figure(1, figsize=[12, 4])

        if title:
            plt.suptitle(title, fontsize=16)

        plt.subplot(1,2,1)
        plt.title("Train data")
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=.5)
        scatter = plt.scatter(X_train[:,0], X_train[:,1], c=y_train, s=80, cmap=cmap, alpha=0.5, label="True")
        plt.scatter(X_train[:,0], X_train[:,1], c=model.predict(X_train), s=20, cmap=cmap, label="Predicted")
        if show_colorbar:
            plt.colorbar()
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)    
        # FIXME: legend_elements is not supported in matplotlib 3.0.3
#         plt.legend(*scatter.legend_elements(), title="Class:")
        plt.legend()
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.title("Test data")
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=.5)
        scatter = plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=80, cmap=cmap, alpha=0.5, label="True")
        plt.scatter(X_test[:,0], X_test[:,1], c=model.predict(X_test), s=20, cmap=cmap, label="Predicted")
        if show_colorbar:
            plt.colorbar()
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)    
        # FIXME: legend_elements is not supported in matplotlib 3.0.3
#         plt.legend(*scatter.legend_elements(), title="Class:")
        plt.legend()
        plt.grid(True)

        plt.show()
        
    @staticmethod
    def show_prediction_plot(model, X, y, title=None, cmap="tab10", proba=False):

        step = 0.01

        x1_min = np.min([X[:,0].min(), X[:,0].min()])
        x1_max = np.max([X[:,0].max(), X[:,0].max()])

        x2_min = np.min([X[:,1].min(), X[:,1].min()])
        x2_max = np.max([X[:,1].max(), X[:,1].max()])

        x1_min = x1_min - (0.1*np.abs(x1_min))
        x1_max = x1_max + (0.1*np.abs(x1_max))
        x2_min = x2_min - (0.1*np.abs(x2_min))
        x2_max = x2_max + (0.1*np.abs(x2_max))

        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, step), 
                             np.arange(x2_min, x2_max, step))
        points = np.c_[xx.ravel(), yy.ravel()]

        if proba is True and hasattr(model, "predict_proba") and len(model.classes_) == 2:
            cmap = cm.bwr
            Z = model.predict_proba(points)[:, 1]
        elif proba is True and hasattr(model, "decision_function") and len(model.classes_) == 2:
            cmap = cm.bwr
            Z = model.decision_function(points)
        else:
            Z = model.predict(points)
        
        Z = Z.reshape(xx.shape)

        plt.figure(1, figsize=[6, 4])

        if title:
            plt.suptitle(title, fontsize=16)

        plt.subplot(1,1,1)
        plt.title("Train data")
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=.5)
        scatter = plt.scatter(X[:,0], X[:,1], c=y, s=80, cmap=cmap, alpha=0.5)
        plt.scatter(X[:,0], X[:,1], c=model.predict(X), s=20, cmap=cmap)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)                   
#         plt.legend(*scatter.legend_elements(), title="Class:")
        plt.grid(True)

        plt.show()

class RPlot:
    """Regression plot class with static methods"""
    
    @staticmethod
    def show_init_data_plot(x, y):
        plt.title("Initial Data")
        plt.plot(x, y, "o", c="g")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    @staticmethod
    def show_train_test_plots(model, X_train, y_train, X_test, y_test, title=None):
    
        plt.figure(1, figsize=[12, 4])

        if title:
            plt.suptitle(title, fontsize=16)
        
        x_min = np.min([X_train.min(), X_test.min()])
        x_max = np.max([X_train.max(), X_test.max()])
        
        x_min = x_min + 0.1*x_min
        x_max = x_max + 0.1*x_max
        
        xx = np.arange(x_min, x_max, 0.01)[:, np.newaxis]
        
        plt.subplot(1,2,1)
        plt.title("Train data")
        plt.plot(X_train, y_train, "o", c="g")
        plt.plot(xx, model.predict(xx), c="g", linewidth=2)
        plt.plot(X_train, model.predict(X_train), "o", color="red", lw=2)
        plt.vlines(X_train, ymin=y_train, ymax=model.predict(X_train), colors="black", linestyles="dotted")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.title("Test data")
        plt.plot(X_test, y_test, "o", c="g")
        plt.plot(xx, model.predict(xx), c="green", label="max_depth=5", linewidth=2)
        plt.plot(X_test, model.predict(X_test), "o", color="red", lw=2)
        plt.vlines(X_test, ymin=y_test, ymax=model.predict(X_test), colors="black", linestyles="dotted")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        plt.show()


show_cplots = CPlot.show_train_test_plots
show_init_cplots = CPlot.show_init_data_plot
show_prediction_cplots = CPlot.show_prediction_plot

show_rplots = RPlot.show_train_test_plots
show_init_rplots = RPlot.show_init_data_plot
