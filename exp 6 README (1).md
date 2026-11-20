# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data=pd.read_csv("fish.csv")


Species	Weight	Length	Diagonal	Height	Width
0	Bream	242.0	25.4	30.0	11.5200	4.0200
1	Bream	290.0	26.3	31.2	12.4800	4.3056
2	Bream	340.0	26.5	31.1	12.3778	4.6961
3	Bream	363.0	29.0	33.5	12.7300	4.4555
4	Bream	430.0	29.0	34.0	12.4440	5.1340
...	...	...	...	...	...	...
154	Smelt	12.2	12.2	13.4	2.0904	1.3936
155	Smelt	13.4	12.4	13.5	2.4300	1.2690
156	Smelt	12.2	13.0	13.8	2.2770	1.2558
157	Smelt	19.7	14.3	15.2	2.8728	2.0672
158	Smelt	19.9	15.0	16.2	2.9322	1.8792
159 rows × 6 columns

df=pd.DataFrame(data)
print("-----Dataset-----")
print(df)
X=df[['Length','Diagonal','Height']]
Y=df['Weight']
model=LinearRegression()
model.fit(X,Y)

print("\n Intercept(a):",model.intercept_)
print("Coefficients (b1,b2,b3):",model.coef_)

print(f"\n Regression Equation:Weight={model.intercept_:.2f}+{model.coef_[0]:.2f}xLength+{model.coef_[1]:.2f}xDiagonal+{model.coef_[2]:.2f}xHeight")


----Dataset-----
    Species  Weight  Length  Diagonal   Height   Width
0     Bream   242.0    25.4      30.0  11.5200  4.0200
1     Bream   290.0    26.3      31.2  12.4800  4.3056
2     Bream   340.0    26.5      31.1  12.3778  4.6961
3     Bream   363.0    29.0      33.5  12.7300  4.4555
4     Bream   430.0    29.0      34.0  12.4440  5.1340
..      ...     ...     ...       ...      ...     ...
154   Smelt    12.2    12.2      13.4   2.0904  1.3936
155   Smelt    13.4    12.4      13.5   2.4300  1.2690
156   Smelt    12.2    13.0      13.8   2.2770  1.2558
157   Smelt    19.7    14.3      15.2   2.8728  2.0672
158   Smelt    19.9    15.0      16.2   2.9322  1.8792

[159 rows x 6 columns]

 Intercept(a): -502.0156658925699
Coefficients (b1,b2,b3): [ 66.35995031 -40.66791791  31.72613524]

 Regression Equation:Weight=-502.02+66.36xLength+-40.67xDiagonal+31.73xHeight

Y_pred=model.predict(X)
print("\nPredicted Weight Values:")
print(Y_pred)

Predicted Weight Values:
[ 328.97461267  370.35415629  384.45052712  463.92134474  434.51371111
  489.24968885  515.69242508  467.37585008  505.66027622  514.41121592
  535.46768711  538.91314539  544.53679697  545.991953    583.10151464
  598.44427364  601.74449854  596.07286305  647.2357104   606.39198264
  627.80872421  667.45537918  665.3615583   660.22402342  688.780219
  676.304325    688.56180741  723.24452779  730.99889399  822.28153839
  831.62836995  837.18946271  871.94397558  943.31408543  886.80965844
  -93.58600858   48.27127853   60.2837799    87.27658385  119.56521937
  136.56785804  134.51968117  121.70626271  137.56002057  180.1408443
  185.32749258  182.19145417  177.43253389  223.49316777  184.36950621
  202.21484357  260.45060224  317.46639652  317.09423119  479.14002486
  322.05150194  323.37508456  381.98308632  513.29918182  676.17439797
  775.27259221   19.69966922   27.61092899  102.26905667  158.84881069
  196.3561546   199.00466225  226.23837674  253.52478731  303.26315914
  370.06077794  404.63612821 -235.46416326  -78.77293479  -35.98235663
   19.22644456   45.85049666   77.3259845   114.9083198   116.13360121
  115.74286008  132.3892835   157.11616251  164.25454294  157.11616251
  186.76239475  196.05385637  181.14257281  196.05385637  188.59821459
  177.41475192  200.84143422  230.53709681  209.06362105  193.46203921
  239.82671729  284.96993916  255.84534701  278.56442153  283.58371514
  306.9204518   345.23425326  406.49835717  374.96769654  371.30015531
  378.63523778  408.53722428  419.05965995  445.290193    610.09463181
  628.4227222   668.15178903  679.2857995   665.94703071  694.73920527
  690.45250566  755.69187246  755.69187246  800.83449155  810.57779576
  796.14340172  820.85756233  777.26635126  877.41398467  891.85393452
  898.04519975  913.79415732  918.91853096  402.81830689  398.0617824
  431.00640932  554.13397672  603.89529646  588.37814776  698.85822196
  684.42283042  735.89124617  752.93593968  817.22449197  844.72127068
  971.18531671 1125.3018513  1181.40550482 1181.40550482 1282.80974015
 -235.73626232 -214.42009669 -215.5125882  -190.40063441 -196.97451956
 -201.80335231 -181.47948604 -181.47948604 -172.57568605 -171.05405899
 -151.07466519 -128.31316907  -80.07788736  -72.40930762]

 plt.figure(figsize=(8,6))
plt.plot(Y.values,label='Actual Weight',marker='o')
plt.plot(Y_pred,label='Predicted Weight',marker='x')
plt.title("Actual vs Predicted Weight(Multiple Regression)")
plt.xlabel("Data Points")
plt.ylabel("Weight")
plt.legend()
plt.grid=(True)
plt.show()

<img width="876" height="637" alt="image" src="https://github.com/user-attachments/assets/d42d06b6-791a-452f-9c11-a8e796792572" />

## Output



## Result
