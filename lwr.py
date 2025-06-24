import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def kernel(point,xmat,k):
    m,n= np.shape(xmat)
    weights=np.matrix(np.eye((m)))

    for j in range(m):
        diff=point-xmat[j]
        weights[j,j]=np.exp(diff*diff.T/(-2*k**2))

    return weights

def localWeight(point,xmat,ymat,k):
    wei=kernel(point,xmat,k)
    # The formula for W in Local Weighted Regression is (X.T * W * X)^-1 * X.T * W * Y
    W=(xmat.T * wei * xmat).I * (xmat.T * wei * ymat.T)
    return W

def localWeightedRegression(xmat, ymat, k):
    m,n=np.shape(xmat)
    ypred=np.zeros(m)
    for i in range(m):
        # Calculate the weight for each point and then predict the y-value
        ypred[i]=xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

def graphPlot(x, ypred, mbill, mtip): # Added mbill and mtip as arguments
    # Sort the x values (total_bill) to ensure the plot is continuous
    # .A1 converts the numpy matrix column to a 1D numpy array for argsort
    sortindex=np.argsort(x[:,1].A1)
    xsort=x[sortindex, 1] # Get the sorted 'Total Bill' values

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    # Plot the sorted total_bill against the corresponding sorted predicted tips
    ax.plot(xsort, ypred[sortindex], color='red', linewidth=5, label='LWR Prediction') # Added label

    # Add scatter plot of original data points
    ax.scatter(mbill.A1, mtip.A1, color='blue', label='Data Points') # Added scatter plot

    plt.xlabel('Total bill')
    plt.ylabel('Tip')
    plt.title('Local Weighted Regression')
    plt.legend() # Added legend to show labels
    plt.show()

# Load the dataset
data=pd.read_csv('dataset/tip.csv')

# Extract bill and tip columns
bill=np.array(data.total_bill)
tip=np.array(data.tip)

# Convert to numpy matrices
mbill=np.matrix(bill)
mtip=np.matrix(tip)

# Get the number of data points
m=np.shape(mbill)[1]

# Create a matrix of ones for the intercept term (bias) in the regression model
one=np.matrix(np.ones(m))

# Stack the ones and the bill amounts horizontally to form the feature matrix X
# X will have two columns: [1, total_bill]
x=np.hstack((one.T,mbill.T))

# Perform Local Weighted Regression with k=1.5
ypred=localWeightedRegression(x,mtip,1.5)

# Plot the results
graphPlot(x,ypred, mbill, mtip) # Passed mbill and mtip to graphPlot
