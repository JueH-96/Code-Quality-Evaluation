def solve():
    import sys
    MOD = 998244353

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])

    # Precompute a helper to count how many numbers k in [0..N] have the i-th bit set
    def count_set_bits_upto(N, i):
        # L = N + 1 total numbers from 0 to N
        L = N + 1
        block_size = 1 << (i + 1)  # 2^(i+1)
        full_blocks = L // block_size
        remainder = L % block_size
        # In each full block, the i-th bit is set exactly 1<<i times
        result = full_blocks * (1 << i)
        # For the leftover remainder, the i-th bit is set if remainder > 2^i
        result += max(0, remainder - (1 << i))
        return result

    answer = 0
    # Sum over each bit i that is set in M
    for i in range(60):
        if M & (1 << i):
            # Add how many numbers in [0..N] set bit i
            answer += count_set_bits_upto(N, i)
    print(answer % MOD)
    
def main():
    solve()

# Call solve() as specified
if __name__ == "__main__":
    main()