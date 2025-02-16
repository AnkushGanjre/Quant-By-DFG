import taipy.gui.builder as tgb
import numpy as np

def CalcMonteCarlo(state):
    S=100               # spot price of underlying
    T=1                 # time period one year
    r=0.07              # discount rate to price call & put option, assuming 7%
    sigma = 0.2         # Volatilty of underlying which is standard deviation, 20%
    Nsimulations = 50000 # No. of simulations
    Nsteps = 250        # No. of days. this is for continous discounting
    K = 100             # strike price

    # Doing Browian motion for daily prices
    dt = T/Nsteps       
    drift = (r-(sigma**2)/2)*dt
    a = sigma*np.sqrt(dt)
    x = np.random.normal(0, 1, (Nsimulations, Nsteps))
    # print(x)        # on y axis we have no. of simulations & on x axis we have no. of day

    Smat = np.zeros((Nsimulations, Nsteps))
    Smat[:, 0]+=S
    
    for i in range(1, Nsteps):
        Smat[:, i] += Smat[:, i-1] * np.exp(drift + a*x[:,i])
    
    # print(Smat)

    # payoff for call
    q = Smat[:, -1]-K
    for i in range(len(q)):
        if q[i]<0:
            q[i]=0
    payoff_call = np.mean(q)
    print("Payoff For Call", payoff_call)

    # payoff for call
    p = K-Smat[:, -1]
    for i in range(len(p)):
        if p[i]<0:
            p[i]=0
    payoff_put = np.mean(p)
    print("Payoff For Put", payoff_put)

    call = payoff_call*np.exp(-r*T)
    put = payoff_put*np.exp(-r*T)

    print("Final Call Price: ", call)
    print("Final Put Price: ", put)


with tgb.Page() as optionPricing_page:
    tgb.text("# **European Option Pricing**", mode="md")
    tgb.text("Work in progress", mode="md")
    # tgb.button("Monte Carlo", class_name="plain", on_action=CalcMonteCarlo)
    tgb.html("br")
    
