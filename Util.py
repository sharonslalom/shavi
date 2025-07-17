
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import math
import numpy as np
import pandas as pd

class Util:
    def __init__(self, data, countryId):
        self.data = data.query('country_id == @countryId').drop(columns = ['country_id'])
        self.data.set_index('year', inplace=True)

        self.originalData = self.data.copy()
        self.hasUsedDifferenciate = False
        self.hasUsedLog = False
        self.isUsingGdpSquared = False
        self.squareCounter = 0
        self.distributeXandY()
        print(f'Util open on {countryId}:')

    def useGdpSquared(self):
        self.squareCounter += 1
        self.data[['gdp_squared_' + str(self.squareCounter + 1)]] = self.data[['gdp']]**(self.squareCounter+1)
        self.isUsingGdpSquared = True
        self.distributeXandY()
        return self

    def distributeXandY(self):
        localDf = pd.DataFrame()
        localDf[['gdp']] = self.data[['gdp']] 
        if self.isUsingGdpSquared:
            for i in range(self.squareCounter):
                localDf[['gdp_squared_' + str(i+2)]] = self.data[['gdp_squared_' + str(i+2)]]
        self.regressors = localDf
        self.regressand = self.data['carbonEmission']

    def print(self):
        print(self.data)
        return self

    def getTurningPoint(self):
        if self.squareCounter == 0: return

        if self.squareCounter == 1:
            # @gpt
            beta_1 = self.olsModel.params['gdp']
            beta_2 = self.olsModel.params['gdp_squared_2']

            # Wendepunkt berechnen
            x_star = -beta_1 / (2 * beta_2)
            print(f"Wendepunkt (Scheitel) bei GDP = {x_star:.2f}")
        
        elif self.squareCounter == 2:
            # @gpt
            b1 = self.olsModel.params['gdp']
            b2 = self.olsModel.params['gdp_squared_2']
            b3 = self.olsModel.params['gdp_squared_3']

            # Diskriminante prüfen
            D = 4*b2**2 - 12*b3*b1

            if D < 0:
                print("Keine realen Wendepunkte (keine N-Form)")
            else:
                x1 = (-2*b2 + np.sqrt(D)) / (6*b3)
                x2 = (-2*b2 - np.sqrt(D)) / (6*b3)
                print(f"Extrempunkte bei GDP = {x1:.2f} und GDP = {x2:.2f}") 
        
        else:
            print('TBI')
        
        return self
    
    def differenciate(self, columns=[]):
        if len(columns) > 0:
            for r in columns:
                print(f'differenziere {r}')
                self.data[[r]] = self.data[[r]].diff()
            self.data = self.data.dropna()
            self.hasUsedDifferenciate = True
            self.distributeXandY()

        return self

    def log(self, columns=[]):
        if len(columns) > 0:
            for r in columns:
                print(f'log {r}')
                self.data[[r]] = np.log(self.data[[r]])
            self.hasUsedLog = True
            self.distributeXandY()

        return self
    
    def correlationsTest(self):
        print(self.data.corr())

        return self
    
    def autoCorrelationTest(self, column = None, lags=3):
        if column != None: 
            plot_acf(self.data[[column]], lags=lags)
        else:  
            plot_acf(self.residuals, lags=lags)
        plt.title(f'ACF {column if column != None else 'residuals'}')
        plt.show()
        return self
    
    def partialAutoCorrelationTest(self):
        plot_pacf(self.residuals, lags=15)
        plt.title(f'PACF residuals')
        plt.show()
        return self
    
    def armaGridSearch(self, r = 5, useConstant = True):
        best_aic = np.inf
        best_order = None
        best_model = None
        for p in range(r):
            for q in range(r):
                try:
                    self.arima(order=(p,0,q), useConstant=useConstant)
                    results = self.arimaModel

                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, q)
                        best_model = results
                except:
                    continue  # z. B. nicht stationär oder instabil

        print(f"Bestes Modell: ARMA{best_order} mit AIC = {best_aic:.2f}")
        print(best_model.summary())
        self.arimaModel = best_model

        return self

    # Augmented Dickey-Fuller (ADF)-Test
    # Nullhypothese H₀: Die Zeitreihe ist nicht stationär.
    # Alternativhypothese H₁: Die Zeitreihe ist stationär.
    # Wenn der p-Wert < 0.05, kannst du H₀ verwerfen → stationär.

    # 'n'	Kein Konstante, kein Trend	Für stationäre Prozesse um Null (selten GDP!)
    # 'c'	Nur Konstante	Für stationäre Prozesse mit Drift
    # 'ct'	Konstante + Trend	Für trendstationäre Reihen (z. B. GDP, Inflation)
    # 'ctt'	Konstante + linearer + quadratischer Trend	sehr selten; für z. B. beschleunigtes Wachstum
    def adfullerTest(self, column = None, regression='c'):
        if column != None: 
            test = adfuller(self.data[[column]].dropna(), regression=regression)
        else:
            test = adfuller(self.residuals, regression=regression)

        p_value = test[1]
        print(f'ADF {column if column != None else 'residuals'} 0 hypothese verwerfen: {p_value < 0.05}, p-wert: {p_value}')
    
        return self
    
    # die Konstante wird gebraucht, 
    # da der Achsenabschnitt sonst bei 0,0 starten wurde.
    def ols(self, useNW = False, useConstant = True): 
        if useConstant:
            self.regressors = sm.add_constant(self.regressors)
        #(Die Standardfehler, die für die Regressionskoeffizienten 
        # angegeben werden, basieren auf der Annahme, dass 
        # der Fehlerterm (𝜀ε) homoskedastisch (gleichmäßige Varianz) und
        #  nicht korreliert ist.)
        localModel = sm.OLS(self.regressand, self.regressors)
        if useNW: 
            self.olsModel = localModel.fit(cov_type='HAC', cov_kwds={'maxlags': self.maxlagsNW()})
        else:
            self.olsModel = localModel.fit()
        
        self.residuals = self.olsModel.resid
        return self

    def maxlagsNW (self): 
        T = len(self.regressors)
        return math.floor(4*(T/100)**(2/9))
    
    def olsSummery(self):
        print(self.olsModel.summary())
        return self
    
    '''
        Wünschenswert:
            Keine Autokorrelation: Ljung Box Test
            Normalverteilung der Fehler: Jarque-Bera
            Homoskedastizität (Konstante Varianz): H-Test

        Modelparameter: 
        ar.L1 Signifikant, da p value < 0.05
        ma.L1 Signifikant, da p value < 0.05
        sigma2 Signifikant, da p value < 0.05
        
        Ljung box (Q): es besteht eine Autokorrelation, p value < 0.05
        Jarque-Bera(JB): Fehler sind nicht normal verteilt, p value < 0.05
        H-test:  Fehler haben keine Konstante Varianz, p < 0.05 

        Prob(*) gibt an mit wie viel Prozent Wahrscheinlichkeit wir die 0 Hypothese erfüllen
    '''

    # AR: Die aktuelle Beobachtung hängt linear von ihren eigenen vergangenen Werten ab.
    # MA: Die aktuelle Beobachtung hängt linear von vergangenen Fehlerwerten (Residuen) ab.
    def arima(self, order=(1,0,1), useConstant = True):
        # Maximum likelyhood ist standard
        if useConstant:
            self.arimaModel = ARIMA(self.residuals, order=order).fit()
        else:
            self.arimaModel = ARIMA(self.residuals, order=order, trend='n').fit()
        return self

    def arimaSummery(self):
        print(self.arimaModel.summary())
        return self

    # Diagramme

    def plot(self, column = None):
        if column != None:
            self.data[column].plot(title='{}-Zeitverlauf'.format(column))
        else:
            self.residuals.plot(title='Residuen-Zeitverlauf')
        
        plt.show()
        return self

    def timePlot (self, column='gdp'):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data[[column]])
        plt.title('{}-Zeitverlauf'.format(column))
        plt.xlabel('Zeit')
        plt.ylabel(column)
        plt.axhline(0, color='red', linestyle='--')
        plt.show()
        return self
    
    def scatterPlot(self, column='gdp'):
        plt.scatter(self.regressors[[column]], self.regressand)
        plt.xlabel(column)
        plt.ylabel("carbon emission")
        plt.show()
        return self
    
    def olsPlotPoly1(self):
        b0, b1, b2 = self.olsModel.params
        x_vals = x_vals = np.linspace(self.data[['gdp']].min(), self.data[['gdp']].max(), 200)
        y_vals = b0 + b1 * x_vals + b2 * x_vals**2

        plt.scatter(self.data[['gdp']], self.data[['carbonEmission']], alpha=0.5, label="Daten")
        plt.plot(x_vals, y_vals, color="red", label="Modellkurve")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
        return self

    def olsPlotPoly2(self):
        b0, b1, b2, b3 = self.olsModel.params
        x_vals = x_vals = np.linspace(self.data[['gdp']].min(), self.data[['gdp']].max(), 200)
        y_vals = b0 + b1 * x_vals + b2 * x_vals**2 + b3 * x_vals**3

        plt.scatter(self.data[['gdp']], self.data[['carbonEmission']], alpha=0.5)
        plt.plot(x_vals, y_vals, color="red", label="Modellkurve")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
        return self
    
    def olsPlotLinear(self):
        b0, b1 = self.olsModel.params

        print(b0, b1)
        

        gdp_line = np.linspace(min(self.regressors['gdp']), max(self.regressors['gdp']), 100)
        carbon_line = b0 + b1 * gdp_line

        plt.scatter(self.regressors[['gdp']], self.regressand)
        plt.plot(gdp_line, carbon_line, color='red')
        plt.show()
        return self

    def olsPlot(self):
        if self.isUsingGdpSquared:
            if self.squareCounter == 1:
                return self.olsPlotPoly1()
            if self.squareCounter == 2:
                return self.olsPlotPoly2()
        else:
            return self.olsPlotLinear()
    # @gpt
    def forecastWithArima(self, steps=5):
        # Zukunfts-X vorbereiten
        last_X = self.regressors.iloc[-1]
        future_X = pd.DataFrame([last_X.copy()] * steps, columns=self.regressors.columns)

        if 'const' in self.olsModel.model.exog_names and 'const' not in future_X.columns:
            future_X = sm.add_constant(future_X)

        # OLS-Vorhersage
        ols_forecast = self.olsModel.predict(future_X)

        # ARIMA-Prognose mit Unsicherheiten
        arima_result = self.arimaModel.get_forecast(steps=steps)
        arima_forecast = arima_result.predicted_mean
        arima_var = arima_result.var_pred_mean  # → das ist die Vorhersagevarianz

        # Gesamtprognose (auf Differenzebene)
        delta_y_forecast = ols_forecast.values + arima_forecast.values

        # Zurück ins Originalniveau
        if self.hasUsedDifferenciate:
            last_y = self.regressand.iloc[-1]
            y_forecast = last_y + np.cumsum(delta_y_forecast)
        else:
            y_forecast = delta_y_forecast

        # Zukunftsindex
        last_index = self.regressand.index[-1]
        future_index = pd.RangeIndex(start=last_index + 1, stop=last_index + 1 + steps)

        assert len(future_index) == len(y_forecast)

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(self.regressand, label='Historisch')
        plt.plot(future_index, y_forecast, label='Forecast (OLS+ARIMA)', color='blue')

        # Konfidenzintervall plotten (vereinfachend nur ARIMA-Unsicherheit, kein OLS-Fehler kombiniert)
        ci_lower = y_forecast - 1.96 * np.sqrt(arima_var)
        ci_upper = y_forecast + 1.96 * np.sqrt(arima_var)

        plt.fill_between(future_index, ci_lower, ci_upper, color='blue', alpha=0.3, label='95% CI (ARIMA)')

        plt.title("Prognose mit Unsicherheit (OLS + ARIMA)")
        plt.xlabel("Jahr")
        plt.ylabel("carbonEmission")
        plt.legend()
        plt.show()

        # Zusätzlich: Varianz tabellarisch ausgeben
        print("\nARIMA-Vorhersagevarianzen pro Zeitschritt:")
        for i, var in enumerate(arima_var):
            print(f"Schritt {i+1}: Varianz = {var:.4f}, Std-Abw = {np.sqrt(var):.4f}")

        return self
 