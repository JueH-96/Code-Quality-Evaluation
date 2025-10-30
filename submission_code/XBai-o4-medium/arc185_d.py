MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    # The formula derived is N * M * (2*N + M - 1) mod MOD
    result = (N * M) % MOD
    temp = (2 * N + M - 1) % MOD
    result = (result * temp) % MOD
    print(result)

if __name__ == "__main__":
    main()