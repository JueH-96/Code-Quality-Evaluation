def count_set_bits(N, b):
    higher = N // (1 << (b + 1))
    remainder = N % (1 << (b + 1))
    return higher * (1 << b) + max(remainder - (1 << b) + 1, 0)

def main():
    MOD = 998244353
    N, M = map(int, input().split())
    total = 0
    for b in range(61):  # Since N and M can be up to 2^60 - 1
        if (M >> b) & 1:
            count = count_set_bits(N, b)
            total = (total + count) % MOD
    print(total)

if __name__ == "__main__":
    main()