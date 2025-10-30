MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    term1 = pow(M-1, N, MOD)
    if N % 2 == 0:
        term2 = (M - 1) % MOD
    else:
        term2 = (- (M - 1)) % MOD
    
    ans = (term1 + term2) % MOD
    print(ans)

if __name__ == '__main__':
    main()