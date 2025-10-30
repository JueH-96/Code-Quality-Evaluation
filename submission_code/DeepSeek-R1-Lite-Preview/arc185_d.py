def main():
    MOD = 998244353
    N, M = map(int, input().split())
    total_nodes = N * M + 1
    expectation = (N * M % MOD) * (total_nodes % MOD) % MOD
    print(expectation)

if __name__ == "__main__":
    main()