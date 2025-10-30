# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    MOD = 998244353
    
    # Calculate the length of N
    L = len(str(N))
    
    # Calculate 10^L % MOD
    ten_power_L = pow(10, L, MOD)
    
    # Calculate (10^(L*N) - 1) % MOD
    ten_power_LN_minus_1 = (pow(ten_power_L, N, MOD) - 1 + MOD) % MOD
    
    # Calculate the modular inverse of (10^L - 1) % MOD
    ten_power_L_minus_1 = (ten_power_L - 1 + MOD) % MOD
    ten_power_L_minus_1_inv = pow(ten_power_L_minus_1, MOD - 2, MOD)
    
    # Calculate the sum of the geometric series modulo MOD
    S = (ten_power_LN_minus_1 * ten_power_L_minus_1_inv) % MOD
    
    # Calculate the result
    result = (N * S) % MOD
    
    # Output the result
    print(result)