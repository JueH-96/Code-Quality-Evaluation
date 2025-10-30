def solve(N, A):
    result = 0
    
    for bit in range(30):  # 10^8 < 2^30, so 30 bits are enough
        # Compute the prefix XOR for this bit
        prefix_bit = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_bit[i] = prefix_bit[i - 1] ^ ((A[i - 1] >> bit) & 1)
        
        # Count the number of ranges with the bit set
        bit_contribution = 0
        count = [0, 0]  # count[0] = number of indices with prefix_bit 0, count[1] = number of indices with prefix_bit 1
        
        # Initialize count for j = 0
        count[prefix_bit[0]] += 1
        
        for j in range(1, N + 1):
            # Count the number of indices i-1 from 0 to j-1 such that prefix_bit[i-1] != prefix_bit[j]
            bit_contribution += count[1 - prefix_bit[j]]
            count[prefix_bit[j]] += 1
        
        result += (1 << bit) * bit_contribution
    
    return result