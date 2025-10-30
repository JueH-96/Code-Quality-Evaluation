MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    
    if N == 1:
        print(1)
        return
    
    inv_N = pow(N, MOD-2, MOD)
    inv_2 = (MOD + 1) // 2  # 499122177
    
    base = ((N - 2) * inv_N) % MOD
    term = pow(base, K, MOD)
    
    part1 = (N + 1) % MOD
    part2 = ((1 - N) % MOD) * term % MOD
    numerator = (part1 + part2) % MOD
    
    e = (numerator * inv_2) % MOD
    print(e)

if __name__ == "__main__":
    main()