# YOUR CODE HERE
import sys

# Function to compute popcount (number of set bits) efficiently.
# It tries to use the built-in int.bit_count() method available in Python 3.10+
# for performance. If not available, it falls back to using bin(x).count('1').
try:
    # Check if int.bit_count method exists. This check works on Python 3.10+.
    _ = (0).bit_count()
    # If it exists, use it.
    popcount_func = lambda x: x.bit_count()
except AttributeError:
    # If int.bit_count is not available (older Python versions),
    # use bin(x).count('1') as a fallback.
    popcount_func = lambda x: bin(x).count('1')
    # For further optimization on older Python versions, one could implement
    # Kernighan's algorithm or use a lookup table if performance is critical.
    # Example Kernighan's algorithm implementation:
    # def countSetBits(n):
    #     count = 0
    #     while (n > 0):
    #         n &= (n-1) # This operation clears the least significant set bit
    #         count+= 1
    #     return count
    # popcount_func = countSetBits


# Implementation of the Fast Walsh-Hadamard Transform (FWHT) specifically for XOR convolution.
# The transform is performed in-place on the input array `A`.
def fwht(A):
    """ Fast Walsh-Hadamard Transform (in-place implementation). """
    N = len(A) # The size of the array, must be a power of 2.
    
    # The transform proceeds in log2(N) stages.
    # `h` represents half the size of the blocks being processed in the current stage.
    h = 1
    while h < N:
        # Iterate through the array, processing blocks of size `h * 2`.
        # `i` is the starting index of each block.
        for i in range(0, N, h * 2):
            # Within each block, iterate through the first half (size `h`).
            # `j` is the index within the first half of the block.
            for j in range(i, i + h):
                # Perform the butterfly operation:
                x = A[j]      # Element from the first half of the current sub-block.
                y = A[j + h]  # Corresponding element from the second half.
                A[j] = x + y      # Update the element in the first half.
                A[j + h] = x - y  # Update the element in the second half.
        
        # Double the step size `h` for the next stage, effectively merging smaller blocks into larger ones.
        h *= 2 

def solve():
    # Read input dimensions H (number of rows) and W (number of columns) from standard input.
    H, W = map(int, sys.stdin.readline().split())
    
    # Calculate the size required for the FWHT arrays. This is 2^W, as there are 2^W possible patterns
    # for a row of length W and 2^W possible column flip combinations.
    N_fft = 1 << W 
    
    # Initialize an array `counts` to store the frequency of each distinct row pattern found in the input grid.
    # The index `k` of the array corresponds to the row pattern represented by the integer `k` (treating the row as a binary number).
    counts = [0] * N_fft
    
    # Read H rows from standard input.
    for _ in range(H):
        # Read a row represented as a string of '0's and '1's.
        row_str = sys.stdin.readline().strip()
        # Convert this binary string representation into an integer mask.
        row_mask = int(row_str, 2)
        # Increment the frequency count for this specific row pattern.
        counts[row_mask] += 1

    # Initialize an array `cost` to store the cost associated with transforming any row pattern `P`
    # into a state `X` by applying column flips `C` (where `X = P XOR C`).
    # The cost for a state `X` is defined as min(popcount(X), W - popcount(X)). This represents the minimum
    # number of '1's achievable for a row that results in state `X` after column flips, by either keeping
    # the row as is or flipping it (Operation X).
    cost = [0] * N_fft
    for i in range(N_fft):
        # Compute the population count (number of set bits) for the integer mask `i`.
        popcount = popcount_func(i)
        # Calculate the cost associated with state `i`.
        cost[i] = min(popcount, W - popcount)

    # Apply the Fast Walsh-Hadamard Transform (FWHT) to both the `counts` array and the `cost` array.
    # This transforms the arrays into the Walsh-Hadamard domain, where the XOR convolution operation
    # becomes a simple element-wise multiplication.
    fwht(counts) # Transform the counts array in-place.
    fwht(cost)   # Transform the cost array in-place.

    # Compute the element-wise product of the transformed arrays.
    # The resulting array `S_hat` stores the FWHT of the desired XOR convolution result `S = counts * cost`.
    # `S[C]` represents the total minimum cost for a fixed column flip pattern `C`.
    S_hat = [0] * N_fft
    for i in range(N_fft):
        S_hat[i] = counts[i] * cost[i]

    # Apply FWHT again to the `S_hat` array. This effectively performs the inverse FWHT.
    # The property of FWHT for XOR convolution is that applying it twice scales the original vector by N:
    # FWHT(FWHT(A)) = N * A.
    # Therefore, after this step, `S_hat[i]` will hold `N_fft` times the actual convolution result `S[i]`.
    fwht(S_hat) 

    # Find the minimum value in the resulting convolution array S.
    # Initialize `min_sum` with a theoretical maximum possible sum (H * W). This serves as a safe upper bound
    # for the minimum sum we are looking for. Handles H=0 case correctly as H*W=0.
    min_sum = H * W 
    
    # Iterate through all possible column flip patterns `C` (represented by the index `i` from 0 to N_fft-1).
    for i in range(N_fft):
        # `S_hat[i]` currently holds `N_fft * S[i]`, where `S[i]` is the minimum total sum of 1s in the grid
        # achievable with the column flip pattern `C=i` (and applying the optimal row flips for that `C`).
        # We perform integer division `//` by `N_fft` to recover the actual sum `S[i]`.
        # The result is guaranteed to be an integer based on the properties of the problem and the transform.
        current_sum = S_hat[i] // N_fft 
        
        # Update the overall minimum sum found so far by comparing it with the sum for the current column flip pattern.
        min_sum = min(min_sum, current_sum)

    # Print the final minimum possible sum to standard output.
    print(min_sum)

# Execute the main logic of the solution by calling the solve function.
solve()