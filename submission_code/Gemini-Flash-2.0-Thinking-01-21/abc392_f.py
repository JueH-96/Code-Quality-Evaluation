# YOUR CODE HERE
import sys

# Function to update the Fenwick tree (add value at index)
def update(bit, idx, val):
    """Adds val to the BIT at index idx (1-based)."""
    # N is the maximum index (inclusive) handled by the BIT
    N = len(bit) - 1
    while idx <= N:
        bit[idx] += val
        idx += idx & -idx

# Function to find the k-th smallest element (index) in the Fenwick tree
# Assumes BIT stores counts (usually non-negative).
# This function finds the smallest index (1-based) i such that the sum up to i is >= k.
# It is optimized using traversal of the BIT structure.
def find_kth(k, N, bit):
    """
    Finds the smallest index (1-based) i such that the prefix sum query(i) >= k.
    Assumes the BIT represents counts of elements at indices.
    """
    idx = 0
    # Find the largest power of 2 less than or equal to N
    # This determines the starting step size for the binary lifting on the BIT
    p = 1
    while p * 2 <= N:
        p *= 2

    # Traverse down the BIT structure from the largest range to smaller ones
    while p > 0:
        # Check the potential next index by adding the current step size 'p'
        next_idx = idx + p
        
        # If the potential index is within bounds (1 to N)
        # and the sum stored at bit[next_idx] (representing a range of indices)
        # is less than the remaining target count 'k'
        if next_idx <= N and bit[next_idx] < k:
            # It means the k-th element is beyond this range (ending at next_idx).
            # Subtract the count of elements in this range from k
            k -= bit[next_idx]
            # Move the current base index past this range
            idx = next_idx
            
        # Move to the next smaller power of 2 (halve the step size)
        p >>= 1

    # After the traversal, 'idx' is the largest index such that query(idx) < original_k.
    # Therefore, the k-th element is located at index idx + 1 (1-based).
    return idx + 1

def solve():
    # Read N from the first line of input
    N = int(sys.stdin.readline())
    # Read the P values (P_1, P_2, ..., P_N) from the second line
    # These are space-separated integers
    P = list(map(int, sys.stdin.readline().split()))

    # Initialize Fenwick tree (BIT) for 1-based indexing, size N+1.
    # bit[i] will store counts for ranges, enabling efficient prefix sum queries.
    # We use it here to track which positions in the final array are "available".
    bit = [0] * (N + 1)

    # Initialize BIT: Mark all N positions (1 to N) as available by adding 1 to each.
    # Initially, all final array slots are available for numbers 1..N.
    for i in range(1, N + 1):
        update(bit, i, 1)

    # Array to store the final result. It will have N elements (0-based indexing).
    result_A = [0] * N

    # Process numbers i from N down to 1.
    # This backward approach allows us to determine the final position of each number.
    # When processing number `i` (from N down to 1), its insertion position P_i
    # is given by P[i-1] in the input list (since P is 0-indexed for i=1..N).
    # P_i represents the 1-based rank among the elements 1..i that will eventually
    # occupy the available slots not taken by i+1, ..., N.
    for i in range(N, 0, -1):
        # P_i is the target rank (1-based) among currently available slots.
        # This is the position relative to the elements 1..i.
        target_rank = P[i - 1]

        # Find the index (1-based) in the final array that corresponds to the
        # target_rank-th available slot among the ones not yet filled by i+1, ..., N.
        final_array_idx_1based = find_kth(target_rank, N, bit)

        # Place the number i at the found position in the result array.
        # Convert the 1-based index from find_kth to a 0-based index for result_A.
        result_A[final_array_idx_1based - 1] = i

        # Mark this position as occupied in the BIT by decreasing its count by 1.
        # This makes this slot unavailable for numbers < i in subsequent steps.
        update(bit, final_array_idx_1based, -1)

    # Output the final array elements separated by spaces.
    print(*result_A)

# Call the solve function to run the program
solve()