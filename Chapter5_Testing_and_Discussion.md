# CHAPTER 5
# TESTING AND DISCUSSION

## 5.1. Test Results

This section presents the data analysis verification process conducted to evaluate the forecasting performance of the ARIMA and SARIMA models across the five selected NASDAQ-100 stocks. The verification follows the rolling one-step-ahead forecasting methodology described in Chapter 4, where each model predicts one trading day at a time, incorporates the actual observed value, and refits before the next prediction. The forecasted values are then compared against the actual closing prices in the test set using three error-based metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).

### 5.1.1. Verification of Stationarity

Prior to model fitting, the Augmented Dickey-Fuller (ADF) test was applied to verify the stationarity assumption required by both ARIMA and SARIMA models. The results confirmed that all five raw closing price series were non-stationary, with p-values ranging from 0.8082 (AMZN) to 0.9981 (GOOGL), all exceeding the 0.05 significance threshold. After applying first-order differencing (d = 1), all five series achieved stationarity with p-values of 0.0000. This verification step ensures that the fundamental assumption underlying both models is satisfied before proceeding with parameter estimation and forecasting.

| Stock | Raw p-value | Result       | Differenced p-value | Result     |
|-------|-------------|--------------|---------------------|------------|
| AAPL  | 0.8424      | Non-stationary | 0.0000            | Stationary |
| NVDA  | 0.9774      | Non-stationary | 0.0000            | Stationary |
| MSFT  | 0.8634      | Non-stationary | 0.0000            | Stationary |
| GOOGL | 0.9981      | Non-stationary | 0.0000            | Stationary |
| AMZN  | 0.8082      | Non-stationary | 0.0000            | Stationary |

*Table 5.1 ADF Stationarity Test Results Summary*

### 5.1.2. Verification of Model Parameter Selection

The auto_arima function from the pmdarima library was used to automatically identify the optimal ARIMA and SARIMA parameters through AIC-based stepwise search. The selected parameters were verified by examining the diagnostic plots (residual analysis, Q-Q plots, and correlogram) to confirm that the model residuals approximate white noise.

| Stock | ARIMA Order | SARIMA Order (p,d,q)(P,D,Q)[m] |
|-------|-------------|-------------------------------|
| AAPL  | (0, 1, 0)   | (0, 1, 0)(0, 0, 0)[5]        |
| NVDA  | (1, 1, 1)   | (2, 1, 3)(0, 0, 1)[5]        |
| MSFT  | (2, 1, 2)   | (0, 1, 0)(0, 0, 0)[5]        |
| GOOGL | (2, 1, 2)   | (0, 1, 0)(0, 0, 0)[5]        |
| AMZN  | (0, 1, 0)   | (0, 1, 0)(0, 0, 0)[5]        |

*Table 5.2 Verified ARIMA and SARIMA Model Orders*

A key observation from the parameter selection is that four out of five stocks (AAPL, MSFT, GOOGL, AMZN) received a seasonal component of (0, 0, 0), indicating that auto_arima found no statistically significant weekly seasonal pattern in these stocks. Only NVDA exhibited a non-trivial seasonal moving average component (Q = 1). This finding is significant for the comparative analysis, as it suggests that the SARIMA model's seasonal extension provides limited additional value for daily stock price forecasting.

### 5.1.3. Forecasting Performance Results

The rolling one-step-ahead forecasting was executed for both ARIMA and SARIMA models across all five stocks. The test set comprised 201 trading days (March 17, 2025 to December 31, 2025) for each stock. The performance metrics are presented below:

| Stock | Model  | RMSE   | MAE    | MAPE (%) |
|-------|--------|--------|--------|----------|
| AAPL  | ARIMA  | 4.2919 | 2.7291 | 1.2343   |
| AAPL  | SARIMA | 4.2919 | 2.7291 | 1.2343   |
| NVDA  | ARIMA  | 3.7892 | 2.8289 | 1.8467   |
| NVDA  | SARIMA | 3.8132 | 2.8645 | 1.8681   |
| MSFT  | ARIMA  | 6.5367 | 4.5457 | 0.9817   |
| MSFT  | SARIMA | 6.4237 | 4.4135 | 0.9527   |
| GOOGL | ARIMA  | 4.3816 | 3.1361 | 1.4763   |
| GOOGL | SARIMA | 4.3277 | 3.1077 | 1.4645   |
| AMZN  | ARIMA  | 4.6694 | 3.2158 | 1.5149   |
| AMZN  | SARIMA | 4.6694 | 3.2158 | 1.5149   |

*Table 5.3 Complete Performance Metrics for All Stocks*

Based on the MAPE criterion, the best-performing model for each stock is identified as follows:

| Stock | Best Model | MAPE Difference |
|-------|-----------|-----------------|
| AAPL  | Tied (ARIMA preferred) | 0.0000 percentage points |
| NVDA  | ARIMA     | 0.0214 percentage points |
| MSFT  | SARIMA    | 0.0290 percentage points |
| GOOGL | SARIMA    | 0.0118 percentage points |
| AMZN  | Tied (ARIMA preferred) | 0.0000 percentage points |

*Table 5.4 Best Model Identification by Stock*

### 5.1.4. MAPE Classification

According to the MAPE interpretation framework proposed by Lewis (1982) and presented in Sub-chapter 2.2.4, all forecasting results across both models and all five stocks fall within the "Highly Accurate Forecasting" category (MAPE < 10%). The highest MAPE observed was 1.8681% (NVDA, SARIMA) and the lowest was 0.9527% (MSFT, SARIMA). This confirms that both ARIMA and SARIMA models are capable of producing highly accurate one-step-ahead forecasts for NASDAQ-100 stock prices.

| MAPE Range | Classification | Stocks in This Range |
|------------|---------------|---------------------|
| < 10%      | Highly accurate forecasting | All 5 stocks (both models) |

*Table 5.5 MAPE Classification Results*

## 5.2. Discussion

This section validates the forecasting results against the evaluation framework established in Sub-chapter 2.4, which states that model evaluation is conducted by comparing the forecasted values generated by the models with the actual observed stock prices, using MAE, RMSE, and MAPE as the primary accuracy metrics.

### 5.2.1. Comparative Analysis of ARIMA vs SARIMA

The central research problem of this study asks: between the ARIMA and SARIMA forecasting method, which model is more accurate for time series forecasting performance towards stock price movement within the NASDAQ 100 index?

Based on the empirical results presented in Table 5.3, the answer is nuanced:

**ARIMA performed better or equal on 3 out of 5 stocks:**
- **AAPL**: ARIMA and SARIMA produced identical results across all three metrics (RMSE: 4.292, MAE: 2.729, MAPE: 1.23%). This occurred because the SARIMA seasonal component was (0,0,0), making the SARIMA model functionally equivalent to ARIMA. In this case, ARIMA is preferred due to its simpler model structure.
- **NVDA**: ARIMA marginally outperformed SARIMA (MAPE: 1.847% vs 1.868%). Interestingly, NVDA was the only stock where auto_arima identified a non-trivial seasonal component (Q = 1), yet this seasonal term did not translate into improved forecasting accuracy. This suggests that the detected seasonal pattern may be a statistical artifact rather than a genuine cyclical structure.
- **AMZN**: Both models produced identical results (RMSE: 4.669, MAE: 3.216, MAPE: 1.515%), again because the SARIMA seasonal component was (0,0,0). ARIMA is preferred for parsimony.

**SARIMA performed better on 2 out of 5 stocks:**
- **MSFT**: SARIMA achieved a lower MAPE of 0.953% compared to ARIMA's 0.982%, a difference of 0.029 percentage points. SARIMA also outperformed on RMSE (6.424 vs 6.537) and MAE (4.414 vs 4.546).
- **GOOGL**: SARIMA achieved a lower MAPE of 1.465% compared to ARIMA's 1.476%, a difference of 0.012 percentage points. SARIMA also showed marginal improvements on RMSE and MAE.

### 5.2.2. Interpretation of the Marginal Differences

A critical observation is that the performance differences between ARIMA and SARIMA are extremely small across all five stocks. The largest MAPE difference was only 0.029 percentage points (MSFT), while two stocks (AAPL and AMZN) showed zero difference. This finding has several important implications:

1. **Absence of weekly seasonality and the Efficient Market Hypothesis**: The trivial seasonal components (0,0,0) identified for four out of five stocks indicate that daily closing prices of NASDAQ-100 stocks do not exhibit statistically significant weekly seasonal patterns. This is consistent with the efficient market hypothesis (EMH), particularly the weak form, which posits that current stock prices already reflect all information contained in historical price data. Under this framework, any recurring seasonal pattern — such as a weekly cycle — would be identified and exploited by market participants, thereby eliminating the pattern through arbitrage. The empirical finding that auto_arima consistently failed to detect meaningful seasonal structures across four major NASDAQ-100 stocks provides supporting evidence for the weak-form EMH in the context of daily stock price data. This also explains why SARIMA, despite its theoretical capability to model seasonality, offered no practical advantage over ARIMA for these stocks.

2. **The random-walk implication of ARIMA(0,1,0)**: A notable finding is that two out of five stocks (AAPL and AMZN) were best modelled by ARIMA(0,1,0), which is mathematically equivalent to a random walk model. Under this specification, the best forecast for tomorrow's price is simply today's price, meaning the model does not identify any exploitable autoregressive or moving average patterns in the differenced series. While this model achieves low MAPE values (1.23% for AAPL and 1.51% for AMZN), the high accuracy is largely a consequence of the one-step-ahead forecasting methodology rather than the model's ability to detect meaningful price patterns. In practical terms, a random walk forecast provides limited actionable information for investment decision-making, as it does not predict the direction or magnitude of price changes. This limitation should be considered when interpreting the forecasting results: a low MAPE does not necessarily imply that the model generates useful trading signals, but rather that stock prices tend to change by small amounts from one day to the next.

3. **SARIMA's additional complexity is not justified**: While SARIMA is theoretically capable of capturing seasonal patterns, the empirical results demonstrate that this additional capability does not translate into meaningful forecasting improvements for daily stock price data. The SARIMA model requires more computational resources and longer fitting times due to the additional seasonal parameters, yet delivers negligible accuracy gains.

4. **Both models achieve highly accurate forecasts**: All MAPE values fall below 2%, well within the "Highly Accurate Forecasting" threshold of 10% defined by Lewis (1982). This confirms that both ARIMA and SARIMA are effective tools for short-term stock price forecasting when applied using a rolling one-step-ahead methodology.

### 5.2.3. Validation Against Evaluation Framework

The evaluation framework in Sub-chapter 2.4 specifies three metrics for assessing model performance:

**MAE Evaluation**: MAE measures the average magnitude of forecasting errors without considering their direction. Across all stocks, MAE values ranged from 2.729 (AAPL) to 4.546 (MSFT). These values represent the average dollar amount by which the forecast deviates from the actual closing price. For investment decision-making purposes, an average error of $2.73 to $4.55 on stocks priced between $150 and $500 represents a relatively small deviation.

**RMSE Evaluation**: RMSE places greater emphasis on larger errors due to the squaring process. The RMSE values ranged from 3.789 (NVDA) to 6.537 (MSFT). The fact that RMSE values are consistently higher than MAE values across all stocks indicates the presence of occasional larger forecast errors, which is expected given the inherent volatility of stock prices. However, the RMSE-to-MAE ratio remains moderate, suggesting that extreme outlier errors are not dominant.

**MAPE Evaluation**: MAPE provides the most intuitive measure of forecast accuracy as a percentage. All MAPE values fall below 2%, indicating that on average, the models' predictions deviate from actual prices by less than 2%. According to the Lewis (1982) classification:
- All 10 model-stock combinations (5 stocks × 2 models) achieved MAPE < 10%, classifying them as "Highly Accurate Forecasting."
- The best individual result was MSFT-SARIMA with MAPE of 0.9527%.
- The worst individual result was NVDA-SARIMA with MAPE of 1.8681%.

### 5.2.4. Validation Against Previous Studies

The results of this study are consistent with findings from previous research discussed in Chapter 2:

- Kurnia et al. (2025) reported that the ARIMA model achieved MAPE values below 10% for stock price forecasting of PT. Bank Central Asia Tbk. The current study confirms this finding, with all ARIMA MAPE values ranging from 0.98% to 1.85% across five NASDAQ-100 stocks.

- Kruba et al. (2025) demonstrated that SARIMA achieved MAPE values below 10% for PT Indofood Sukses Makmur Tbk stock data. Similarly, this study found SARIMA MAPE values ranging from 0.95% to 1.87%, confirming the model's effectiveness.

- Wang, Li, and Lim (2021) highlighted that ARIMA and SARIMA models may not always be sufficient for modeling complex or highly volatile financial time series. While the current study found both models to be highly accurate for one-step-ahead forecasting, the trivial seasonal components suggest that these models may indeed have limitations in capturing more complex patterns beyond simple trend and short-term autocorrelation.

### 5.2.5. Implications for Investment Decision-Making

The objective of this study, as stated in Sub-chapter 1.3, is to evaluate the performance of the ARIMA and SARIMA models in forecasting stock price movements and to examine the potential utilization of the forecasting results as information in investment decision-making.

Based on the results:

1. **Both models are viable for short-term forecasting**: With MAPE values consistently below 2%, both ARIMA and SARIMA can provide reliable one-step-ahead price estimates that may assist investors in anticipating next-day price movements.

2. **ARIMA is recommended as the primary model**: Given that ARIMA performed better or equal on 3 out of 5 stocks, requires fewer parameters, is computationally less expensive, and the performance differences with SARIMA are negligible, ARIMA represents the more practical choice for daily stock price forecasting in the NASDAQ-100 index.

3. **Limitations as decision support**: While the forecasting accuracy is high, investors should be aware that these models rely solely on historical price patterns and do not incorporate external factors such as political conditions, government policies, macroeconomic news, or qualitative market sentiment, as noted in the scope of study (Sub-chapter 1.5). Therefore, the forecasting results should be used as one component of a broader investment analysis framework, not as the sole basis for trading decisions.

### 5.2.6. Summary of Findings

The following table summarizes the key findings of this study:

| Finding | Description |
|---------|-------------|
| Best overall model | ARIMA (preferred on 3/5 stocks; simpler and equally effective) |
| Performance difference | Marginal (max 0.029 percentage points MAPE difference) |
| Seasonal patterns | Not significant for 4/5 stocks at weekly frequency |
| Forecast accuracy | Highly accurate (all MAPE < 2%, well below 10% threshold) |
| Practical recommendation | ARIMA preferred for daily NASDAQ-100 stock forecasting |

*Table 5.6 Summary of Key Findings*
