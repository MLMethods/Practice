import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_one_tailed_right_tdistribution(xlim, value, df):
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.t.pdf(x, df=df)
    pr = stats.t.cdf(value, df=df)
    plt.axvspan(value, xlim[1], color="g", alpha=0.1)
    plt.plot(x, y, lw=2, color="green")
    plt.title("t-distribution")
    plt.xlim(xlim)
    plt.fill_between(x, 0, y, where=x>value, facecolor="green", alpha=0.5)
    plt.axvline(value, color="black", linestyle="--")
    plt.annotate("$1-F_{one}(%0.2f)=%0.2f$" % (value, 1-pr), fontsize=14, xycoords="data", xytext=(-1.5, 0.1), xy=(-2.6, 0.01))
    plt.grid(True)

    
def plot_one_tailed_left_tdistribution(xlim, value, df):
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.t.pdf(x, df=df)
    pr = stats.t.cdf(value, df=df)
    plt.axvspan(xlim[0], value, color="g", alpha=0.1)
    plt.plot(x, y, lw=2, color="green")
    plt.title("t-distribution")
    plt.xlim(xlim)
    plt.fill_between(x, 0, y, where=x<=value, facecolor="green", alpha=0.5)
    plt.axvline(value, color="black", linestyle="--")
    plt.annotate("$F_{one}(%0.2f)=%0.2f$" % (value, pr), fontsize=14, xycoords="data", xytext=(-1.5, 0.1), xy=(-2.6, 0.01))
    plt.grid(True)

    
def plot_two_tailed_tdistribution(xlim, value, df):
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.t.pdf(x, df=df)   
    pr = 2* stats.t.cdf(-abs(value), df=df)
    plt.plot(x, y, lw=2, color="green")
    plt.axvspan(xlim[0], -value, color="g", alpha=0.1)
    plt.axvspan(value, xlim[1], color="g", alpha=0.1)
    plt.title("t-distribution")
    plt.xlim(xlim)
    plt.fill_between(x, 0, y, where=y<stats.t.pdf(value, df=df), facecolor="green", alpha=0.5)
    plt.axvline(-value, color="black", linestyle="--")
    plt.axvline(value, color="black", linestyle="--")
    plt.annotate("$F_{two}(%0.2f)=%0.2f$" % (value, pr), fontsize=14, xy=(-2.6, 0.01), xycoords="data", xytext=(-1.5, 0.1))
    plt.grid(True)
    
    
def plot_one_tailed_right_normal_distribution(xlim, value):
    x = np.linspace(xlim[0], xlim[1], 1000)    
    y = stats.norm.pdf(x, loc=0, scale=1)
    pr = stats.norm.cdf(value, loc=0, scale=1)
    plt.axvspan(value, xlim[1], color="g", alpha=0.1)
    plt.plot(x, y, lw=2, color="green")
    plt.title("Standard Normal Distribution")
    plt.xlim(xlim)
    plt.fill_between(x, 0, y, where=x>value, facecolor="green", alpha=0.5)
    plt.axvline(value, color="black", linestyle="--")
    plt.annotate("$1-\\Phi_N({:.2f})={:.2f}$".format(value, 1-pr), (0.35,0.4), 
                 fontsize=14, xycoords="axes fraction")
    plt.grid(True)
    
    
def plot_one_tailed_left_normal_distribution(xlim, value):
    x = np.linspace(xlim[0], xlim[1], 1000)    
    y = stats.norm.pdf(x, loc=0, scale=1)
    pr = stats.norm.cdf(value, loc=0, scale=1)
    plt.axvspan( xlim[0], value, color="g", alpha=0.1)
    plt.plot(x, y, lw=2, color="green")
    plt.title("Standard Normal Distribution")
    plt.xlim(xlim)
    plt.fill_between(x, 0, y, where=x<=value, facecolor="green", alpha=0.5)
    plt.axvline(value, color="black", linestyle="--")
    plt.annotate("$\\Phi_N({:.2f})={:.2f}$".format(value, pr), (0.35,0.4), 
                 fontsize=14, xycoords="axes fraction")
    plt.grid(True)
    
    
def plot_two_tailed_normal_distribution(xlim, value):
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.norm(loc=0, scale=1).pdf(x) 
    pr = stats.norm(loc=0, scale=1).cdf(value) - stats.norm(loc=0, scale=1).cdf(-value)
    plt.axvspan(-value, value, color="g", alpha=0.1)
    plt.plot(x, y, lw=2, color="green")
    plt.title("Standard Normal Distribution")
    plt.xlim(xlim)
    plt.fill_between(x, 0, y, where=y>stats.norm(loc=0, scale=1).pdf(value), facecolor="green", alpha=0.5)
    plt.axvline(-value, color="black", linestyle="--")
    plt.axvline(value, color="black", linestyle="--")
    plt.annotate("$\\gamma({:.2f})={:.2f}$".format(value, pr), (0.35,0.4), 
                     fontsize=14, xycoords="axes fraction") 
    plt.grid(True)


def plot_two_tailed_outside_normal_distribution(xlim, value):
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.norm(loc=0, scale=1).pdf(x) 
    pr = stats.norm(loc=0, scale=1).cdf(-abs(value))
    plt.axvspan(xlim[0], -value, color="g", alpha=0.1)
    plt.axvspan(value, xlim[1], color="g", alpha=0.1)
    plt.plot(x, y, lw=2, color="green")
    plt.title("Standard Normal Distribution")
    plt.xlim(xlim)
    plt.fill_between(x, 0, y, where=y<stats.norm(loc=0, scale=1).pdf(value), facecolor="green", alpha=0.5)
    plt.axvline(-value, color="black", linestyle="--")
    plt.axvline(value, color="black", linestyle="--")
    plt.annotate("$2\\cdot(1-\\Phi({:.2f})={:.2f}$".format(value, pr), (0.35,0.4), 
                     fontsize=14, xycoords="axes fraction") 
    plt.grid(True)

    
def get_z_by_alpha_for_two_tailed(alpha):
    return stats.norm.ppf(alpha/2, loc=0, scale=1)


def get_t_by_alpha_for_two_tailed(alpha, df):
    return stats.t.ppf(1 - alpha/2, df) 


def get_pvalue_for_two_tails_norm(z):
    return 2 * (stats.norm.cdf(-abs(z), loc=0, scale=1))


def get_pvalue_for_two_tails_tdistribtion(t, df):
    return 2 * stats.t.cdf(-abs(t), df)


def get_z(x, mu, se):
    return (x - mu) / se


def plot_two_tailed_pvalue_for_standard_norm(z, alpha=0.05, xlim=(-4,4)):

    z_alpha_lower = get_z_by_alpha_for_two_tailed(alpha)
    z_alpha_upper = -z_alpha_lower

    z_upper = abs(z)
    z_lower = -z_upper

    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.norm.pdf(x, loc=0, scale=1)
    pvalue = get_pvalue_for_two_tails_norm(z)

    plt.title("Standard Normal Distribution")

    zorder = 1
    plt.plot(x, y, lw=2, color="green", zorder=zorder)
    plt.xlim(xlim)

    zorder += 1
    plt.fill_between(x, 0, y, where=x<=z_lower, facecolor="red", alpha=0.5, zorder=zorder+1 if z_upper > z_alpha_upper else zorder)
    plt.fill_between(x, 0, y, where=x>=z_upper, facecolor="red", alpha=0.5, zorder=zorder+1 if z_upper > z_alpha_upper else zorder)

    plt.fill_between(x, 0, y, where=x<=z_alpha_lower, facecolor="green", alpha=1, zorder=zorder if z_upper > z_alpha_upper else zorder+1)
    plt.fill_between(x, 0, y, where=x>=z_alpha_upper, facecolor="green", alpha=1, zorder=zorder if z_upper > z_alpha_upper else zorder+1)

    zorder += 2
    plt.axvline(z_alpha_lower, color="g", linestyle="--", zorder=zorder)
    plt.axvline(z_alpha_upper, color="g", linestyle="--", zorder=zorder)
    plt.axvline(z_lower, color="r", linestyle="--", zorder=zorder)
    plt.axvline(z_upper, color="r", linestyle="--", zorder=zorder)

    zorder += 1
    plt.annotate("$\\alpha$", fontsize=14, xy=(z_alpha_lower, 0.004), xycoords="data", xytext=(-0.1, 0.2),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), zorder=zorder)
    plt.annotate("", fontsize=14, xy=(z_alpha_upper, 0.004), xycoords="data",
                               xytext=(0.1, 0.19), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), zorder=zorder)

    plt.annotate("p-value", fontsize=14, xy=(z_lower, 0.004), xycoords="data", xytext=(-0.7, 0.1),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), zorder=zorder)
    plt.annotate("", fontsize=14, xy=(z_upper, 0.004), xycoords="data",
                               xytext=(0.1, 0.09), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), zorder=zorder)

    plt.annotate("$z$", fontsize=14, xy=(z, 0), xycoords="data",
                               xytext=(z, -0.08), zorder=zorder)

    plt.annotate("$z_{\\alpha}$", fontsize=14, xy=(z_alpha_upper if z > 0 else z_alpha_lower, 0), xycoords="data",
                               xytext=(z_alpha_upper if z > 0 else z_alpha_lower, -0.08), zorder=zorder)

    zorder += 1
    plt.annotate("$\\alpha={:0.2f}$".format(alpha), (0.67,0.9), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("p-value=${:0.4f}$".format(pvalue), (0.67,0.8), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("$z={:0.2f}$".format(z), (0.67,0.7), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("$z_{\\alpha}=%0.2f$"%(z_alpha_upper if z > 0 else z_alpha_lower), (0.67,0.6), fontsize=14, xycoords="axes fraction", zorder=zorder)

    plt.grid(True)
  
    
def plot_two_tailed_pvalue_for_norm(x_bar, mu=0, se=1, alpha=0.05, xlim=(-4,4)):

    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.norm.pdf(x, loc=mu, scale=se)

    pvalue = get_pvalue_for_two_tails_norm(get_z(x_bar, mu, se))
    z_alpha = abs(get_z_by_alpha_for_two_tailed(alpha))

    upper_bound = x_bar if x_bar > mu else 2*mu - x_bar
    lower_bound = 2*mu - x_bar if x_bar > mu else x_bar

    x_alpha_upper = mu + z_alpha*se
    x_alpha_lower = mu - z_alpha*se
    
    plt.title("Normal Distribution with $\mu$ and $SE$")

    zorder = 1
    plt.plot(x, y, lw=2, color="green", zorder=zorder)
    plt.xlim(xlim)

    zorder += 1
    plt.fill_between(x, 0, y, where=x<=lower_bound, facecolor="red", alpha=0.5, zorder=zorder+1 if alpha > pvalue else zorder)
    plt.fill_between(x, 0, y, where=x>=upper_bound, facecolor="red", alpha=0.5, zorder=zorder+1 if alpha > pvalue else zorder)

    plt.fill_between(x, 0, y, where=x<=x_alpha_lower, facecolor="green", alpha=1, zorder=zorder if alpha > pvalue else zorder+1)
    plt.fill_between(x, 0, y, where=x>=x_alpha_upper, facecolor="green", alpha=1, zorder=zorder if alpha > pvalue else zorder+1)

    zorder += 2
    plt.axvline(x_alpha_upper, color="g", linestyle="--", zorder=zorder)
    plt.axvline(x_alpha_lower, color="g", linestyle="--", zorder=zorder)

    plt.axvline(lower_bound, color="r", linestyle="--", zorder=zorder)
    plt.axvline(upper_bound, color="r", linestyle="--", zorder=zorder)

    zorder += 1

    # TODO: Add p-value and alpha arrows
#     plt.annotate("$\\alpha$", fontsize=14, xy=(z_alpha_lower, 0.005), xycoords="data", xytext=(19, 0.06),
#                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#     plt.annotate("", fontsize=14, xy=(z_alpha_upper, 0.005), xycoords="data",
#                                xytext=(21, 0.057), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

#     plt.annotate("p-value", fontsize=14, xy=(lower_bound, 0.005), xycoords="data", xytext=(17, 0.03),
#                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#     plt.annotate("", fontsize=14, xy=(upper_bound, 0.005), xycoords="data",
#                                xytext=(22, 0.027), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    plt.annotate("$\\mu$", fontsize=14, xy=(mu, 0), xycoords="data",
                               xytext=(mu, 0), zorder=zorder)

    plt.annotate("$\\bar{x}$", fontsize=14, xy=(x_bar, 0), xycoords="data",
                               xytext=(x_bar, 0), zorder=zorder)

    plt.annotate("$\\bar{x}_{\\alpha}$", fontsize=14, xy=(x_alpha_upper if x_bar > mu else x_alpha_lower, 0), xycoords="data",
                               xytext=(x_alpha_upper if x_bar > mu else x_alpha_lower, 0), zorder=zorder)

    zorder += 1
    plt.annotate("$\\alpha={:0.2f}$".format(alpha), (0.67,0.9), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("p-value=${:0.4f}$".format(pvalue), (0.67,0.8), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("$\\bar{x}=%0.2f$" % (x_bar), (0.67,0.7), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("$\\bar{x}_{\\alpha}=%0.2f$" % (x_alpha_upper if x_bar > mu else x_alpha_lower), (0.67,0.6), fontsize=14, xycoords="axes fraction", zorder=zorder)
        
    plt.grid(True)


def plot_two_tailed_pvalue_for_tdistribution(t, df, alpha=0.05, xlim=(-4,4)):

    t_alpha_upper = get_t_by_alpha_for_two_tailed(alpha, df)
    t_alpha_lower = -t_alpha_upper

    t_upper = abs(t)
    t_lower = -t_upper

    x = np.linspace(xlim[0], xlim[1], 1000)
    y = stats.norm.pdf(x, loc=0, scale=1)
    pvalue = get_pvalue_for_two_tails_tdistribtion(t, df)

    plt.title("t-Distribution")

    zorder = 1
    plt.plot(x, y, lw=2, color="green", zorder=zorder)
    plt.xlim(xlim)

    zorder += 1
    plt.fill_between(x, 0, y, where=x<=t_lower, facecolor="red", alpha=0.5, zorder=zorder+1 if t_upper > t_alpha_upper else zorder)
    plt.fill_between(x, 0, y, where=x>=t_upper, facecolor="red", alpha=0.5, zorder=zorder+1 if t_upper > t_alpha_upper else zorder)

    plt.fill_between(x, 0, y, where=x<=t_alpha_lower, facecolor="green", alpha=1, zorder=zorder if t_upper > t_alpha_upper else zorder+1)
    plt.fill_between(x, 0, y, where=x>=t_alpha_upper, facecolor="green", alpha=1, zorder=zorder if t_upper > t_alpha_upper else zorder+1)

    zorder += 2
    plt.axvline(t_alpha_lower, color="g", linestyle="--", zorder=zorder)
    plt.axvline(t_alpha_upper, color="g", linestyle="--", zorder=zorder)
    plt.axvline(t_lower, color="r", linestyle="--", zorder=zorder)
    plt.axvline(t_upper, color="r", linestyle="--", zorder=zorder)

    zorder += 1
    plt.annotate("$t$", fontsize=14, xy=(t, 0), xycoords="data",
                               xytext=(t, -0.08), zorder=zorder)

    plt.annotate("$t_{\\alpha}$", fontsize=14, xy=(t_alpha_upper if t > 0 else t_alpha_lower, 0), xycoords="data",
                               xytext=(t_alpha_upper if t > 0 else t_alpha_lower, -0.08), zorder=zorder)

    zorder += 1
    plt.annotate("$\\alpha={:0.2f}$".format(alpha), (0.67,0.9), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("p-value=${:0.4f}$".format(pvalue), (0.67,0.8), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("$t={:0.2f}$".format(t), (0.67,0.7), fontsize=14, xycoords="axes fraction", zorder=zorder)
    plt.annotate("$t_{\\alpha}=%0.2f$"%(t_alpha_upper if t > 0 else t_alpha_lower), (0.67,0.6), fontsize=14, xycoords="axes fraction", zorder=zorder)

    plt.grid(True)

if __name__ == "__main__":
    pass
