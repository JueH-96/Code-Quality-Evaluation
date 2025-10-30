# YOUR CODE HERE
def main():
    import sys
    MOD = 998244353
    N, M = map(int, sys.stdin.read().split())
    total = 0
    for i in range(60):
        if (M >> i) & 1:
            period = 1 << (i + 1)
            n = N + 1
            full_periods = n // period
            remainder = n % period
            cnt = full_periods * (1 << i) + max(0, remainder - (1 << i))
            total += cnt
    print(total % MOD)

if __name__ == "__main__":
    main()