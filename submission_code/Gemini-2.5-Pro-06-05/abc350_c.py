import sys

def solve():
    """
    Solves the permutation sorting problem.

    Reads a permutation from standard input, sorts it using a series of swaps,
    and prints the number of swaps and the swap operations to standard output.
    The algorithm is an optimized selection sort with O(N) time complexity.
    """
    try:
        # Use fast I/O
        input = sys.stdin.readline
        
        # Read N and handle potential empty lines in some environments
        n_str = input()
        if not n_str.strip():
            return
        N = int(n_str)
        
        # Read the permutation A
        A = list(map(int, input().split()))
    except (ValueError, IndexError):
        # Gracefully handle malformed input or empty lines
        return

    # pos[v] stores the 0-based index of value v.
    # The list is 1-indexed for values, so its size is N+1.
    pos = [0] * (N + 1)
    for i, val in enumerate(A):
        pos[val] = i

    # List to store the swap operations
    swaps = []

    # Iterate through the array to place each element in its correct position.
    for i in range(N):
        # The correct value for index `i` is `i + 1`.
        if A[i] != i + 1:
            # Get the value that is currently at the target position i.
            val_at_i = A[i]
            # The value that should be at this position.
            correct_val = i + 1
            
            # Find the 0-based index of the correct value using the pos array.
            idx_of_correct = pos[correct_val]
            
            # Perform the swap on the array A.
            A[i], A[idx_of_correct] = A[idx_of_correct], A[i]
            
            # Update the positions in the pos array to reflect the swap.
            pos[val_at_i] = idx_of_correct
            pos[correct_val] = i
            
            # Record the swap using 1-based indices for the output.
            swaps.append((i + 1, idx_of_correct + 1))

    # Print the total number of swaps.
    print(len(swaps))
    # Print each swap operation.
    for i, j in swaps:
        print(i, j)

# Run the solution
solve()