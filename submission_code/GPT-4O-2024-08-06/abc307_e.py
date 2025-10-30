def solve():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, M = map(int, data.split())
    
    MOD = 998244353
    
    def mod_exp(base, exp, mod):
        result = 1
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp //= 2
        return result
    
    if M == 1:
        # If M is 1, there's only one color, and it's impossible to color the circle
        # without two adjacent people having the same color.
        print(0)
        return
    
    # Calculate M * (M-1)^(N-1) % MOD
    term1 = M * mod_exp(M-1, N-1, MOD) % MOD
    
    # Calculate (-1)^N * (M-1) % MOD
    term2 = ((-1) ** N) * (M-1) % MOD
    
    # Result is (term1 + term2) % MOD
    result = (term1 + term2) % MOD
    
    print(result)