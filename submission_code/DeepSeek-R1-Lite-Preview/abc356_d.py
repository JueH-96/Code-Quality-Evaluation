def main():
    MOD = 998244353
    N, M = map(int, input().split())
    S = 0
    for i in range(60):
        if M & (1 << i):
            higher = N // (1 << (i + 1))
            count = higher * (1 << i)
            remainder = N % (1 << (i + 1))
            if remainder >= (1 << i):
                count += remainder - (1 << i) + 1
            S = (S + count) % MOD
    print(S)

if __name__ == "__main__":
    main()