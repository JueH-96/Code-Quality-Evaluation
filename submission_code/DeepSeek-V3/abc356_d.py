# YOUR CODE HERE
def main():
    MOD = 998244353
    N, M = map(int, input().split())
    
    # Precompute the number of set bits in M
    popcount_M = bin(M).count('1')
    
    # Initialize the result
    result = 0
    
    # Iterate over each bit position
    for bit in range(60):  # Since M can be up to 2^60 - 1
        if (M >> bit) & 1:
            # Calculate the number of numbers <= N where the bit is set
            # The pattern is: every 2^(bit+1) numbers, 2^bit numbers have the bit set
            full_cycles = (N + 1) // (1 << (bit + 1))
            remainder = (N + 1) % (1 << (bit + 1))
            count = full_cycles * (1 << bit)
            count += max(0, remainder - (1 << bit))
            result += count
            if result >= MOD:
                result -= MOD
    
    print(result % MOD)

if __name__ == "__main__":
    main()