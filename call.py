import math
from scipy.stats import norm # type: ignore


def main():
    S = 52.25
    X = 48
    r = 0.057
    T = 7/12
    sigma = math.sqrt(0.12)
    call_price = calculate_call_price(S, X, r, T, sigma)
    print(f"The price of the call option is: {call_price}")
    
    put_price = calculate_put_price(S, X, r, T, sigma)
    print(f"The price of the put option is: {put_price}")
    
    verify_put_call_parity(call_price, put_price, S, X, r, T)

def calculate_call_price(S, X, r, T, sigma):
    """
    Calculate the price of a European call option using the Black-Scholes model.

    Args:
    S (float): Current stock price
    X (float): Strike price
    r (float): Risk-free interest rate (annual rate, expressed as a decimal)
    T (float): Time to expiration (in years)
    sigma (float): annual volatility (continuous compounding)
    """
    
    d1 = (math.log(S/X) + (r + (sigma**2)/2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    c = S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)

    return c

def calculate_put_price(S, X, r, T, sigma):
    """
    Calculate the price of a European put option using the Black-Scholes model.
    """
    d1 = (math.log(S/X) + (r + (sigma**2)/2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    put_price = X * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price

def verify_put_call_parity(call_price, put_price, S, X, r, T):
    """
    Verify the put-call parity relationship.
    """
    parity_check = call_price + X * math.exp(-r * T) - S
    print(f"Put-call parity check: {parity_check}")
    print(f"Difference: {abs(parity_check - put_price)}")

if __name__ == "__main__":
    main()