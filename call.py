import math
from scipy.stats import norm # type: ignore


def main():
    S = 42.35
    X = 42
    r = 0.038
    T = 6/12
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
    sigma (float): Annual volatility (continuous compounding)

    Returns:
    float: The price of the European call option

    Operations:
    1. Calculate d1 and d2 parameters
    2. Use the cumulative distribution function (CDF) of the standard normal distribution
    3. Apply the Black-Scholes formula to calculate the call option price
    """
    
    d1 = (math.log(S/X) + (r + (sigma**2)/2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    c = S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)

    return c

def calculate_put_price(S, X, r, T, sigma):
    """
    Calculate the price of a European put option using the Black-Scholes model.

    Args:
    S (float): Current stock price
    X (float): Strike price
    r (float): Risk-free interest rate (annual rate, expressed as a decimal)
    T (float): Time to expiration (in years)
    sigma (float): Annual volatility (continuous compounding)

    Returns:
    float: The price of the European put option

    Operations:
    1. Calculate d1 and d2 parameters
    2. Use the cumulative distribution function (CDF) of the standard normal distribution
    3. Apply the Black-Scholes formula to calculate the put option price
    """
    d1 = (math.log(S/X) + (r + (sigma**2)/2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    put_price = X * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price

def verify_put_call_parity(call_price, put_price, S, X, r, T):
    """
    Verify the put-call parity relationship for European options.

    Args:
    call_price (float): Price of the call option
    put_price (float): Price of the put option
    S (float): Current stock price
    X (float): Strike price
    r (float): Risk-free interest rate (annual rate, expressed as a decimal)
    T (float): Time to expiration (in years)

    Returns:
    None

    Operations:
    1. Calculate the put-call parity relationship
    2. Print the result of the parity check
    3. Print the absolute difference between the left and right sides of the parity equation
    """
    parity_check = call_price - put_price - (S - X * math.exp(-r * T))
    print(f"Put-call parity check: {parity_check}")
    print(f"Difference: {abs(parity_check)}")

if __name__ == "__main__":
    main()