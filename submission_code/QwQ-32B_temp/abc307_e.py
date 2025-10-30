MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    
    a_prev, b_prev = 1, 0
    for k in range(2, N+1):
        a_current = b_prev
        term1 = a_prev * (M-1) % MOD
        term2 = b_prev * (M-2) % MOD
        b_current = (term1 + term2) % MOD
        a_prev, b_prev = a_current, b_current
    
    h = (M * a_prev) % MOD
    g = M * pow(M-1, N-1, MOD) % MOD
    ans = (g - h) % MOD
    print(ans)
    
if __name__ == "__main__":
    main()