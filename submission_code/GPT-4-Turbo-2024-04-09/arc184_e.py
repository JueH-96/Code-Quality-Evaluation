def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    sequences = []
    for _ in range(N):
        sequences.append([int(data[idx + i]) for i in range(M)])
        idx += M
    
    # To calculate f(i, j) efficiently, we need to understand the operation:
    # The operation on A_i is essentially computing the prefix XOR of the sequence.
    # After M operations, any sequence becomes its prefix XOR sequence.
    # Thus, if two sequences have the same prefix XOR after any number of operations,
    # they will become identical after at most M operations.
    
    # Compute prefix XOR for each sequence
    prefix_xor_sequences = []
    for seq in sequences:
        pxor = []
        current_xor = 0
        for num in seq:
            current_xor ^= num
            pxor.append(current_xor)
        prefix_xor_sequences.append(tuple(pxor))
    
    # Dictionary to count occurrences of each prefix XOR sequence
    from collections import defaultdict
    prefix_count = defaultdict(int)
    for pxor in prefix_xor_sequences:
        prefix_count[pxor] += 1
    
    # Calculate the result
    result = 0
    for count in prefix_count.values():
        # Each pair (i, j) with i <= j contributes f(i, j) = 0
        result += count * (count + 1) // 2
        result %= MOD
    
    # Output the result
    print(result)

if __name__ == "__main__":
    main()