MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    N, K = map(int, input().split())
    if N == 1:
        print(1)
        return
    
    inv_N = pow(N, MOD-2, MOD)
    E = 1  # E[0]
    for _ in range(K):
        E = (( ( (N - 2) * E ) % MOD + (N + 1) ) % MOD) * inv_N % MOD
    print(E)

if __name__ == '__main__':
    main()