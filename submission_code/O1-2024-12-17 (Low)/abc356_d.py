def main():
    import sys
    mod = 998244353
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    # Function to count how many numbers in [0..N] have the i-th bit set
    def count_bit_set_upto(N, i):
        # Size of one full cycle (bit i is 0 for 2^i numbers, then 1 for 2^i numbers)
        block_size = 1 << (i + 1)
        # Number of complete cycles in [0..N]
        full_cycles = (N + 1) // block_size
        # Remainder beyond complete cycles
        remainder = (N + 1) % block_size
        # Within each full cycle, bit i is set exactly 2^i times
        count_in_full_cycles = full_cycles * (1 << i)
        # In the remainder, bit i is set if remainder > 2^i
        count_in_remainder = max(0, remainder - (1 << i))
        return count_in_full_cycles + count_in_remainder

    ans = 0
    # Sum popcount(k & M) from k=0..N
    # popcount(k & M) counts how many bits i are set in both k and M
    # For each bit i that is set in M, add how many k in [0..N] have i-th bit set
    for i in range(60):
        if (M >> i) & 1:  # if i-th bit of M is 1
            ans = (ans + count_bit_set_upto(N, i)) % mod

    print(ans % mod)

# Don't forget to call main()
if __name__ == "__main__":
    main()