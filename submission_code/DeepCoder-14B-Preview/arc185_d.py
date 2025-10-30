mod = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    
    inv6 = pow(6, mod - 2, mod)
    
    n = N % mod
    m = M % mod
    
    term1 = (n * m) % mod
    term2 = ((m + 1) % mod)
    
    # Compute term3 = (3*N*M - 2*M + 2) mod mod
    part1 = (3 * n) % mod
    part1 = (part1 * m) % mod
    part2 = (2 * m) % mod
    term3 = (part1 - part2) % mod
    term3 = (term3 + 2) % mod
    if term3 < 0:
        term3 += mod
    
    sum_val = term1 * term2 % mod
    sum_val = sum_val * term3 % mod
    sum_val = sum_val * inv6 % mod
    
    print(sum_val)

if __name__ == "__main__":
    main()