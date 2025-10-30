import sys

# Read input
N, K = map(int, sys.stdin.readline().split())
R = list(map(int, sys.stdin.readline().split()))

# Initialize sequence list
sequence = [0] * N

# Recursive function
# index: the current position in the sequence we are filling (from 0 to N-1)
# current_sum: the sum of elements sequence[0]...sequence[index-1]
def solve(index, current_sum):
    # Base case: If we have filled all N elements
    if index == N:
        # Check if the total sum is a multiple of K
        if current_sum % K == 0:
            # If it is, print the sequence
            # print(*sequence) is Python 3 syntax to print list elements space-separated
            print(*sequence)
        return # Backtrack

    # Recursive step: If we still need to fill elements
    # Iterate through possible values for the current element sequence[index]
    # The values can range from 1 to R[index] (inclusive)
    for v in range(1, R[index] + 1):
        # Place the value v at the current index
        sequence[index] = v
        # Recurse for the next index (index + 1) with the updated sum
        solve(index + 1, current_sum + v)

# Start the recursion from the first element (index 0) with an initial sum of 0
solve(0, 0)