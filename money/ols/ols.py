import statsmodels.api as sm
import matplotlib.pyplot as plt

Y = [4.4567, 5.77, 5.9787, 7.3317, 7.3182, 6.5844, 7.8182, 7.8351, 11.0223, 10.6738, 10.8361, 13.615, 13.531]
X = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

est = sm.OLS(Y, sm.add_constant(X)).fit()


print(est.params)
print(est.summary())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X, Y, c='b')
plt.show()
