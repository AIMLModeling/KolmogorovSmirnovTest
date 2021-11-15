import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from numpy.random import seed
mu = 5
sigma = 1
#set seed (e.g. make this example reproducible)
seed(0)
step=2
if(step==1):
    fig, ax = plt.subplots()
    arr_1_more = np.random.normal(mu, sigma, size=6000)
    arr_1_sorted = np.sort(arr_1_more)
    y_cdf = st.norm.cdf(arr_1_sorted, mu, sigma) # the normal cdf
    y_cdf = st.norm.cdf(arr_1_sorted, mu, sigma) # the normal cdf
    norm_data=st.norm.pdf(arr_1_sorted, mu, sigma)
    poisson_pmf = st.poisson.pmf(mu, arr_1_sorted)
    ax.plot(arr_1_sorted, norm_data, label="Normal")
    ax.plot(arr_1_sorted, poisson_pmf, label="Poisson")
    ax.set_xlabel("x", size=12)
    ax.set_ylabel("F(x)", size=12)
    legend = ax.legend(loc="upper left")
    plt.show()
#generate random aray
if(step>1):
    mysize=100
    arr_1_few = np.random.normal(mu, sigma, size=mysize)
    #sort generated array
    arr_1_sorted = np.sort(arr_1_few)
    #edf
    arr_1_edf_few = np.arange(1, len(arr_1_few)+1) / len(arr_1_few)
    #generate dataset that follow a Poisson distribution with mean=mu
    poisson_cdf = 1-st.poisson.cdf(mu, arr_1_sorted)
    fig, ax = plt.subplots()
    ax.plot(arr_1_sorted, arr_1_edf_few, label="Observation")
    y_cdf = st.norm.cdf(arr_1_sorted, mu, sigma) # the normal cdf
    norm_data=st.norm.pdf(arr_1_sorted, mu, sigma)
    ax.plot(arr_1_sorted, y_cdf, label="Theory")
    ax.plot(arr_1_sorted, poisson_cdf, label="Poisson")
    ax.set_xlabel("x", size=12)
    ax.set_ylabel("F(x)", size=12)
    legend = ax.legend(loc="upper left")
    plt.show()
    #calculate absolute difference
    arr_dif_abs = np.abs(y_cdf-arr_1_edf_few)
    print("Difference relative to normal distribution")
    print(arr_dif_abs)
    #get max difference
    dn_ks = max(arr_dif_abs)
    print("Maximum difference relative to normal distribution:{}".format(dn_ks))
    #calculate absolute difference for poisson distribution
    arr_dif_abs_poisson = np.abs(y_cdf-poisson_cdf)
    print("Difference relative to poisson distribution")
    print(arr_dif_abs_poisson)
    ps_ks = max(arr_dif_abs_poisson)
    print("Maximum difference relative to normal distribution:{}".format(dn_ks))
    print("Maximum difference relative to poisson distribution:{}".format(ps_ks))
    D_critial=1.36 * np.sqrt(2/mysize)
    print("Critical value at 95% confidence interval:{}".format(D_critial))