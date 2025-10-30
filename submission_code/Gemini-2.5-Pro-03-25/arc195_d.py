# YOUR CODE HERE
import sys
from collections import deque

# Function to solve a single test case
def solve():
    """
    Solves a single test case for the minimum operations problem.
    Reads input N and sequence A.
    Calculates the minimum total operations (swaps + deletions) required 
    to make the sequence A empty.
    Prints the result to stdout.
    """
    N = int(sys.stdin.readline())
    
    # Read the sequence A based on N
    if N == 0:
        A = [] # Empty sequence
    else:
        line = sys.stdin.readline().split()
        # Check if line is empty just in case, although constraints N>=2 guarantee non-empty line
        if line: 
             A = [int(x) for x in line]
        else: # Should not happen based on problem constraints
             A = []
    
    # Handle base cases N=0 and N=1. Problem constraints state N >= 2,
    # so these cases are technically not reachable based on constraints,
    # but including them makes the function robust.
    if N <= 1: 
        print(N) # For N=0 cost is 0, for N=1 cost is 1 (one deletion)
        return

    # Calculate the initial number of blocks.
    # A block is a maximal contiguous subsequence of identical elements.
    # Without swaps, the minimum number of deletions required is equal to the number of blocks.
    blocks = 0
    if N > 0: # Check if sequence is not empty
        blocks = 1 # A non-empty sequence has at least one block
        # Iterate through adjacent pairs to count boundaries between different elements
        for i in range(N - 1):
            if A[i] != A[i+1]:
                blocks += 1
    
    # Initialize the minimum cost to the number of blocks (cost without swaps)
    current_cost = blocks

    # `check_indices` deque stores indices `k` where a swap might be beneficial.
    # `k` is the 0-based index of the first element in the pair (A[k], A[k+1]) to be swapped.
    # A swap is beneficial if it follows the pattern (x, y, x, y) which becomes (x, x, y, y) after swap,
    # reducing the number of blocks by 2 at the cost of 1 swap operation (net cost reduction of 1).
    check_indices = deque()
    # `in_queue` set tracks indices currently in the deque to prevent duplicate checks.
    in_queue = set()

    # Initial Scan: Find all indices `k` where the beneficial swap pattern exists.
    # The pattern involves elements at indices k-1, k, k+1, k+2.
    # For the pattern check to be valid, k must be in the range [1, N-3].
    for k in range(1, N - 2): 
        # Check pattern condition: A[k-1] == A[k+1] (x == x) and A[k] == A[k+2] (y == y)
        if A[k-1] == A[k+1] and A[k] == A[k+2]:
             # Ensure the elements to swap A[k] and A[k+1] are different.
             # If A[k] == A[k+1], the pattern is (x, x, x, x). Swapping identical elements is useless.
             if A[k] != A[k+1]: 
                 # If index k is not already scheduled for checking, add it to the deque and set.
                 if k not in in_queue:
                     check_indices.append(k)
                     in_queue.add(k)

    # Iteratively apply beneficial swaps until no more such swaps can be found.
    while check_indices:
        k = check_indices.popleft() # Get the next index to check from the left of the deque
        in_queue.remove(k) # Remove from tracking set as it's now being processed

        # Check index bounds for k: 1 <= k <= N-3.
        # This check is mostly for safety against potential logic errors or edge cases with small N.
        if not (1 <= k <= N - 3): 
             continue

        # Re-check the pattern at index k because the array A might have been modified by previous swaps.
        if A[k-1] == A[k+1] and A[k] == A[k+2]:
            # Check again if elements A[k] and A[k+1] are different
            if A[k] != A[k+1]: 
                # Perform the swap A[k] <-> A[k+1]
                A[k], A[k+1] = A[k+1], A[k]
                # This swap reduces the number of blocks by 2, but costs 1 swap operation. Net cost reduction is 1.
                current_cost -= 1 

                # After swap at k, the sequence changes locally. This might create new beneficial swap opportunities
                # or invalidate existing ones nearby. We need to check indices `j` near `k`.
                # The relevant indices `j` for pattern checks are those potentially affected by the change at k, k+1.
                # These are indices k-2, k-1, k, k+1, k+2.
                # Check potential swap centers `j` in range [max(1, k-2), min(N-3, k+2)]
                for j in range(max(1, k - 2), min(N - 2, k + 3)): # Correct range covers indices up to k+2
                    # Check bounds for potential swap center j: 1 <= j <= N-3
                    if 1 <= j <= N - 3: 
                        # Check if the beneficial swap pattern holds at index j
                        if A[j-1] == A[j+1] and A[j] == A[j+2]:
                             # Ensure swap elements are different
                             if A[j] != A[j+1]: 
                                 # If pattern holds and j is not already scheduled for checking, add it.
                                 if j not in in_queue:
                                     check_indices.append(j)
                                     in_queue.add(j)
            # else case: A[k] == A[k+1]. The pattern is (x, x, x, x). Swap does nothing useful. Don't decrement cost or check neighbors.
        # else case: The pattern does not hold at index k anymore (possibly due to prior swaps). Do nothing.
    
    # After the loop finishes, current_cost holds the minimum total operations.
    print(current_cost)


# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
    solve()