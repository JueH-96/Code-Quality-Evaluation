import sys
from math import comb

def main():
    mod = 998244353
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Compute prefix sums S[0..N]
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = (S[i-1] + A[i-1]) % mod
    
    # Precompute coefficients
    coeff = [0] * (K + 1)
    for k in range(K + 1):
        c = comb(K, k)
        d = K - k
        sign = -1 if d % 2 else 1
        coeff[k] = (c * sign) % mod
    
    # Initialize previous sums
    prev_sums = [0] * (K + 1)
    ans = 0
    
    for j in range(N + 1):
        s = S[j]
        # Compute current powers
        pow_j = [0] * (K + 1)
        pow_j[0] = 1 % mod
        for m in range(1, K + 1):
            pow_j[m] = (pow_j[m-1] * s) % mod
        
        # Accumulate contributions
        for k in range(K + 1):
            d = K - k
            temp = (coeff[k] * pow_j[k]) % mod
            temp = (temp * prev_sums[d]) % mod
            ans = (ans + temp) % mod
        
        # Update prev_sums
        for m in range(K + 1):
            prev_sums[m] = (prev_sums[m] + pow_j[m]) % mod
    
    print(ans % mod)

if __name__ == "__main__":
    main()