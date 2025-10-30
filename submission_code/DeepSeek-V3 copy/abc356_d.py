MOD = 998244353

def main():
    N, M = map(int, input().split())
    total = 0
    # Iterate over each bit position
    for bit in range(60):
        if (M >> bit) & 1:
            # Calculate the number of k where the bit is set in k
            # The bit is set in k if k has the bit set
            # The number of k <= N with the bit set is (N + 1) // (1 << (bit + 1)) * (1 << bit) + max(0, (N + 1) % (1 << (bit + 1)) - (1 << bit))
            full_cycles = (N + 1) // (1 << (bit + 1))
            remainder = (N + 1) % (1 << (bit + 1))
            count = full_cycles * (1 << bit) + max(0, remainder - (1 << bit))
            total += count
    print(total % MOD)

if __name__ == "__main__":
    main()