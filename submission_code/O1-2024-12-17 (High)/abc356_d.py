def main():
    import sys
    MOD = 998244353

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    # Counts how many integers k in [0..n] have their i-th bit as 1
    def count_set_bits(n, i):
        block = 1 << (i + 1)         # Size of each repeating pattern
        full_blocks = (n + 1) // block
        leftover = (n + 1) % block

        # In each full block of length 2^(i+1), the bit i is set exactly 2^i times
        ret = full_blocks * (1 << i)

        # Count any leftover range after the last full block
        if leftover > (1 << i):
            ret += leftover - (1 << i)

        return ret

    # Sum up the counts of set bits in (k & M) across all k in [0..N]
    # popcount(k & M) = number of bits i such that M's i-th bit = 1 and k's i-th bit = 1
    # Hence for each bit i where M has a 1, we add how many k's have that bit i set.
    ans = 0
    for i in range(60):  # 60 bits are enough for numbers < 2^60
        if (M >> i) & 1:
            ans += count_set_bits(N, i)

    print(ans % MOD)

# Don't forget to call main!
if __name__ == "__main__":
    main()