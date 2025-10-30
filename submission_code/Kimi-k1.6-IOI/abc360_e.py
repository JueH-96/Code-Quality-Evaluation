MOD = 998244353

def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    if K == 0:
        print(1)
        return
    inv_N = pow(N, MOD-2, MOD)
    e = 1
    for _ in range(K):
        e = (( (N - 2) * e + (N + 1) ) % MOD) * inv_N % MOD
    print(e)

if __name__ == "__main__":
    main()