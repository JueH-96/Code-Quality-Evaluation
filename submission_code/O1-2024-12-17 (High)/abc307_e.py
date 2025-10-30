def main():
    import sys
    mod = 998244353
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    # We need to count the number of ways to assign integers in [0, M-1]
    # to N people in a circle such that no two adjacent people have the same integer.
    #
    # A known result (chromatic polynomial for cycle graph) is that the count
    # equals (M-1)^N + (M-1)*(-1)^N (all computations taken modulo 998244353).
    
    base = (M - 1) % mod
    # Compute (M-1)^N mod 998244353 efficiently:
    p = pow(base, N, mod)
    
    if N % 2 == 0:
        # For even N, we add (M-1)
        ans = (p + base) % mod
    else:
        # For odd N, we subtract (M-1)
        ans = (p - base) % mod
    
    print(ans % mod)

# Do not forget to call main().
if __name__ == "__main__":
    main()