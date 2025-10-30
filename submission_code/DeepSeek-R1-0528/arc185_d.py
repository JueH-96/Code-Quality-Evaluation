mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    
    term1 = (1 + M * (N - 1)) % mod
    part1 = (N % mod) * (M % mod) % mod * term1 % mod
    
    A = (1 + (N - 1) * M) % mod
    
    T1 = M * (M - 1) // 2
    T2 = M * M * (M - 1) // 2
    T3 = M * (M - 1) * (2 * M - 1) // 6
    
    T1 %= mod
    T2 %= mod
    T3 %= mod
    
    S = A * T1 % mod
    S = (S + T2) % mod
    S = (S - T3) % mod
    if S < 0:
        S += mod
    
    part2 = (N % mod) * S % mod
    
    ans = (part1 + part2) % mod
    print(ans)

if __name__ == '__main__':
    main()