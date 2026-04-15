# Residual diagnostics for Change D

## Linearity
Residuals were inspected visually against fitted values and log(size). The cloud remains centred around zero without a strong systematic curve, although dispersion remains broad, so the linear functional form is not obviously contradicted by the plots.

## Homoscedasticity
Breusch-Pagan tests the null of constant residual variance. The LM statistic is 47.819 with p-value 0.0005; the F statistic is 2.491 with p-value 0.0003. Conclusion: Reject homoscedasticity at the 5% level.

## Multicollinearity
Variance inflation factors assess whether predictors are excessively collinear. The maximum observed VIF is 155.403. At least one predictor exceeds the VIF > 10 threshold.

Top VIF values:

```
                                             predictor        vif  flag
     log_total_size_sqm:C(primary_asset_type)[T.Hotel] 155.402727  True
                        C(primary_asset_type)[T.Hotel] 154.307696  True
log_total_size_sqm:C(primary_asset_type)[T.Industrial]  95.420178  True
                   C(primary_asset_type)[T.Industrial]  89.484030  True
    log_total_size_sqm:C(primary_asset_type)[T.Retail]  78.485164  True
```

## Independence of residuals
Durbin-Watson is reported for completeness given the rolling-origin design. The statistic is 1.983. Conclusion: The statistic is close to 2, which is broadly consistent with low residual autocorrelation.

## Normality
Jarque-Bera tests the null that residuals are normally distributed. The statistic is 0.414 with p-value 0.8130. Conclusion: Fail to reject normality at the 5% level.
