def main():
    import sys
    mod = 998244353
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    # Counts how many integers k in [0..n] have the i-th bit set (0-based indexing of bits)
    def count_bit(n, i):
        block_size = 1 << (i + 1)
        full_blocks = (n + 1) // block_size
        remainder = (n + 1) % block_size
        
        # Each full block contributes exactly 2^i set bits in position i
        base = full_blocks * (1 << i)
        
        # Add the partial block's contribution if it exceeds 2^i
        threshold = 1 << i
        add = 0
        if remainder > threshold:
            add = remainder - threshold
        
        return base + add
    
    ans = 0
    # For each bit position, if M has that bit set to 1, count how many k's have that bit set
    for i in range(60):
        if ((M >> i) & 1) == 1:
            ans += count_bit(N, i)
            ans %= mod
    
    print(ans % mod)

# Call main at the end
if __name__ == "__main__":
    main()