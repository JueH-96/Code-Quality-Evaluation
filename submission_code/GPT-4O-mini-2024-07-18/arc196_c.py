def count_strongly_connected_partitions(N, S):
    MOD = 998244353
    
    # Count the number of white and black vertices
    W_count = S.count('W')
    B_count = S.count('B')
    
    # If counts of W and B are not equal, return 0
    if W_count != N or B_count != N:
        return 0
    
    # To ensure strong connectivity, we need to check the prefix sums
    # of the counts of W and B
    white_prefix = 0
    black_prefix = 0
    
    for char in S:
        if char == 'W':
            white_prefix += 1
        else:
            black_prefix += 1
        
        # At any point, the number of black vertices must not exceed
        # the number of white vertices, otherwise we can't form a valid pairing
        if black_prefix > white_prefix:
            return 0
    
    # The number of valid pairings is given by the Catalan number C(N)
    # C(N) = (2N)! / ((N+1)! * N!)
    
    # Precompute factorials and modular inverses
    fact = [1] * (2 * N + 1)
    for i in range(2, 2 * N + 1):
        fact[i] = fact[i - 1] * i % MOD
    
    def mod_inverse(x):
        return pow(x, MOD - 2, MOD)
    
    inv_fact_N = mod_inverse(fact[N])
    inv_fact_N_plus_1 = mod_inverse(fact[N + 1])
    
    # Calculate the Catalan number C(N) modulo MOD
    catalan_number = (fact[2 * N] * inv_fact_N % MOD * inv_fact_N_plus_1 % MOD) % MOD
    
    return catalan_number

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    S = data[1]
    
    result = count_strongly_connected_partitions(N, S)
    print(result)

if __name__ == "__main__":
    main()