# YOUR CODE HERE
import sys

def count_divisors_1_to_M(M):
    """Count the number of divisors for integers 1 to M."""
    divisors = [0] * (M + 1)
    for i in range(1, M + 1):
        for j in range(i, M + 1, i):
            divisors[j] += 1
    return divisors

def compute_sum_of_scores(N, M):
    MOD = 998244353
    
    # Count divisors for numbers 1 to M
    divisors = count_divisors_1_to_M(M)
    
    # Initialize the sum of divisors for sequences of length 1
    sum_for_length_1 = sum(divisors[1:M+1])
    
    # Calculate the geometric sum formula
    # For sequences of length 1: sum_for_length_1
    # For all sequences of length 1 to N: sum_for_length_1 * (1 - M^N) / (1 - M) (if M != 1)
    
    # Handle the case where M = 1 separately
    if M == 1:
        return (N * divisors[1]) % MOD
    
    # Calculate M^N % MOD using fast exponentiation
    M_to_N = pow(M, N, MOD)
    
    # Using the formula for the sum of a geometric series:
    # sum = a * (1 - r^n) / (1 - r)
    # where a = sum_for_length_1, r = M, n = N
    
    # Calculate (1 - M^N) % MOD
    one_minus_M_to_N = (1 - M_to_N) % MOD
    
    # Calculate (1 - M) % MOD
    one_minus_M = (1 - M) % MOD
    
    # Calculate the modular multiplicative inverse of (1 - M)
    inv_one_minus_M = pow(one_minus_M, MOD - 2, MOD)
    
    # Calculate the final result
    result = (sum_for_length_1 * one_minus_M_to_N * inv_one_minus_M) % MOD
    
    return result

if __name__ == "__main__":
    N, M = map(int, input().split())
    print(compute_sum_of_scores(N, M))