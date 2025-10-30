def main():
    import sys
    mod = 998244353
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data)
    
    # Formula for the number of ways to assign integers
    # around a circle so that no two adjacent are the same:
    # (M - 1)^N + (M - 1) * (-1)^N  (all modulo 998244353)
    
    # Compute (M-1)^N mod 998244353
    p = pow(M - 1, N, mod)
    
    # Compute the second part depending on parity of N
    if N % 2 == 0:
        # even
        ans = (p + (M - 1)) % mod
    else:
        # odd
        ans = (p - (M - 1)) % mod
    
    print(ans)

# Do not forget to call main()
main()