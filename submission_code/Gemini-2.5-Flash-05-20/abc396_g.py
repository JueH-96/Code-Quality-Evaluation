import sys

# Function to perform Fast Walsh-Hadamard Transform (FWHT) for XOR convolution
def fwht(a, inverse):
    n = len(a)
    
    # Iterate log2(n) times, where h is the current block size
    h = 1
    while h < n:
        # Iterate over blocks of size 2*h
        for i in range(0, n, h * 2):
            # Iterate within each block
            for j in range(h):
                # Apply the butterfly operation for XOR convolution
                u = a[i + j]
                v = a[i + h + j]
                a[i + j] = u + v
                a[i + h + j] = u - v
        h *= 2 # Double the block size for the next iteration
    
    # If it's an inverse transform, divide by n
    if inverse:
        for i in range(n):
            a[i] /= n

def solve():
    # Read H and W
    H, W = map(int, sys.stdin.readline().split())

    # N is the size of the arrays, which is 2^W
    N = 1 << W

    # Step 1: Initialize frequency array for row configurations
    # freq[k] will store the number of times row k appears in the input grid
    # Use floats for FWHT calculations to handle divisions correctly
    freq = [0.0] * N
    for _ in range(H):
        s = sys.stdin.readline().strip()
        # Convert binary string to integer
        row_val = int(s, 2)
        freq[row_val] += 1.0

    # Step 2: Initialize G array with costs
    # G[k] will store min(popcount(k), W - popcount(k))
    G = [0.0] * N
    for k in range(N):
        # Calculate popcount (number of set bits)
        # Using bin(k).count('1') for general compatibility
        # For Python 3.10+, k.bit_count() is faster
        k_bit_count = bin(k).count('1')
        G[k] = min(k_bit_count, W - k_bit_count)

    # Step 3: Perform FWHT on freq and G arrays
    fwht(freq, False)
    fwht(G, False)

    # Step 4: Element-wise product of the transformed arrays
    # This corresponds to the convolution in the frequency domain
    for k in range(N):
        freq[k] *= G[k]

    # Step 5: Perform inverse FWHT to get the result in the original domain
    # The freq array now contains ans[mask] = sum(freq[k] * G[k ^ mask])
    fwht(freq, True)

    # Step 6: Find the minimum value in the result array
    min_total_sum = float('inf')
    for val in freq:
        # The values might be floats due to division in inverse FWHT.
        # Round to the nearest integer as the sum of bits must be an integer.
        min_total_sum = min(min_total_sum, int(round(val)))

    print(min_total_sum)

# Call the solve function
solve()