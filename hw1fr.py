import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(
    x, y, independent, learningRate=0.01, epochs=1000, m=0, b=0, loss_threshold=1e-4
):
    n = float(len(x))

    loss_y = np.zeros(epochs)

    previous_loss = float("inf")

    for i in range(epochs):
        predictedY = m * x + b
        dM = (-2 / n) * np.sum(x * (y - predictedY))
        dB = (-2 / n) * np.sum(y - predictedY)
        m = m - learningRate * dM
        b = b - learningRate * dB
        totalError = np.sum((y - predictedY) ** 2) / (2 * n)
        loss_y[i] = totalError

        if i > 0 and abs(loss_y[i] - previous_loss) < loss_threshold:
            print(f"stopping early at epoch {i+1} due to loss convergence.")
            epochs = i + 1
            break
        previous_loss = loss_y[i]

    print(f"y = {m}x + {b}")

    plotData(x, y, xtitle=independent, slope=m, intercept=b)
    plt.plot(np.arange(1, epochs + 1, 1), loss_y[:epochs], "r")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Loss over {epochs} epochs")
    plt.show()

    #print(f"final mean squared error loss: {totalError}")

    return m, b, loss_y[:epochs]

def multiplegradientDescent(
    X, y, learningRate=0.01, epochs=994, theta=None, loss_threshold=1e-3
):
    if theta is None:
        theta = np.zeros(X.shape[1])

    n = float(len(y))

    loss_y = np.zeros(epochs)

    previous_loss = float("inf")

    for i in range(epochs):
        predictedY = np.dot(X, theta)
        error = predictedY - y
        gradient = (1 / n) * np.dot(X.T, error)
        theta -= learningRate * gradient
        loss_y[i] = np.mean(error**2)

        if i > 0 and abs(loss_y[i] - previous_loss) < loss_threshold:
            print(f"stopping early at epoch {i+1} due to loss convergence.")
            actual_epochs = i + 1
            break

        previous_loss = loss_y[i]

    print(f"theta values: {theta}")

    loss_space = np.arange(1, actual_epochs + 1, 1)

    plt.plot(loss_space, loss_y[:actual_epochs])
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Loss over {actual_epochs} epochs")
    plt.show()

    final_error = np.mean((np.dot(X, theta) - y) ** 2)
    print(f"final mean squared error loss: {final_error}")

    return theta, loss_y[:actual_epochs]

def plotData(x, y, xtitle, slope=None, intercept=None):
    plt.scatter(x, y)

    if slope is None and intercept is None:
        return

    bestFitY = slope * x + intercept
    plt.plot(x, bestFitY, "m")
    plt.xlabel(xtitle)
    plt.ylabel("y")
    plt.title(f"{xtitle} vs y")
    plt.show()

def predictResults(independent, theta):
    ans = 1
    for x, y in zip(independent, theta[1:]):
        ans += x * y
    return ans + theta[0]

if __name__ == "__main__":
 def main():
    data = np.loadtxt("./D3.csv", delimiter=",", skiprows=1)

    X = data[:, :3]
    y = data[:, 3]

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    X_with_bias = np.column_stack((np.ones(X.shape[0]), X))

    learning_rates = [0.1, 0.07, 0.01]  # diff learning rates

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}\n")

        # using each explanatory variable in isolation (Problem 1)
        for i in range(X.shape[1]):
            x = X[:, i]
            print(f"X{i+1} Regression:")
            m, b, loss = gradientDescent(x, y, f"X{i+1}", learningRate=lr)
            print(f"Final model parameters: m = {m}, b = {b}")
            print(f"Final mean squared error loss: {loss[-1]}\n")

        # using all three explanatory variables (Problem 2)
        print("Using all explanatory variables combined:")
        theta, loss = multiplegradientDescent(X_with_bias, y, learningRate=lr)
        print(f"Final model parameters (theta): {theta}")
        print(f"Final mean squared error loss: {loss[-1]}\n")

        # predict future results
        #print("Predictions for new input values:")
        prediction1 = predictResults((1, 1, 1), theta)
        prediction2 = predictResults((2, 0, 4), theta)
        prediction3 = predictResults((3, 2, 1), theta)
        print(
        f"Predicted Values:\n(1,1,1): {prediction1}\n(2,0,4): {prediction2}\n(3,2,1): {prediction3}"
    )

main()