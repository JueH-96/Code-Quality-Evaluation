def main():
    import sys
    input = sys.stdin.read
    MOD = 998244353
    
    N = int(input().strip())
    
    # Precompute factorials and powers of 10
    factorial = [1] * (N + 1)
    pow10 = [1] * (N + 1)
    
    for i in range(1, N + 1):
        factorial[i] = factorial[i - 1] * i % MOD
        pow10[i] = pow10[i - 1] * 10 % MOD
    
    # Precompute lengths of numbers 1 to N
    lengths = [0] * (N + 1)
    for i in range(1, N + 1):
        lengths[i] = len(str(i))
    
    # Precompute power of 10 sums for each length
    pow10_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        pow10_sum[i] = (pow10_sum[i - 1] + pow10[lengths[i]]) % MOD
    
    # Calculate the sum of all f(P) for permutations P
    result = 0
    for i in range(1, N + 1):
        # Each number i contributes to the result in two ways:
        # 1. As the first part of the number (with its full length)
        # 2. As the subsequent part of the number (with its length as power of 10)
        len_i = lengths[i]
        pow10_len_i = pow10[len_i]
        
        # Contribution when i is at the start of the permutation
        # It will be i * 10^(length of the rest of the numbers)
        # There are (N-1)! permutations where i is at the start
        rest_pow10 = (pow10_sum[N] - pow10_len_i) % MOD
        contribution1 = i * pow10[rest_pow10] % MOD * factorial[N-1] % MOD
        
        # Contribution when i is not at the start
        # It will be i * 10^length of numbers before it
        # There are (N-1)! permutations for each position of i
        contribution2 = i * (pow10_sum[N-1] * factorial[N-1] % MOD) % MOD
        
        result = (result + contribution1 + contribution2) % MOD
    
    print(result)