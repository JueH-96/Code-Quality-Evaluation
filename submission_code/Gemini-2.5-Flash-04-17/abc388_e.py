# YOUR CODE HERE
import sys

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# The mochi sizes are already sorted in ascending order.

# Use a two-pointer approach.
# top_ptr will iterate through potential top mochi from the left (smaller sizes).
# bottom_ptr will iterate through potential bottom mochi from the left,
# searching for a suitable match for the current top_ptr.
# We want to find the maximum number of pairs (top, bottom) such that top <= bottom / 2,
# using distinct mochi.
# A greedy strategy: for the current smallest available mochi (A[top_ptr]),
# find the smallest available mochi (A[bottom_ptr]) that is larger than A[top_ptr]
# and satisfies the condition A[top_ptr] * 2 <= A[bottom_ptr].

top_ptr = 0  # Pointer for the current potential top mochi
bottom_ptr = 0 # Pointer for the current potential bottom mochi
count = 0    # Counter for the number of kagamimochi formed

# Iterate through the array with top_ptr.
# bottom_ptr will search ahead to find a suitable bottom.
while top_ptr < N and bottom_ptr < N:
    # The bottom mochi A[bottom_ptr] must be a distinct mochi from the top A[top_ptr].
    # Its index must be greater than the top mochi's index.
    # The bottom_ptr should always be at least top_ptr + 1 to ensure this.
    # Also, if bottom_ptr has lagged behind top_ptr (e.g., after previous top_ptr increments),
    # we need to advance bottom_ptr to start searching from after the current top_ptr.
    if bottom_ptr <= top_ptr:
        bottom_ptr = top_ptr + 1

    # If bottom_ptr goes out of bounds after adjustment, we can't find any more bottoms.
    if bottom_ptr >= N:
        break

    # Now we have bottom_ptr > top_ptr.
    # Check if A[top_ptr] can be placed on A[bottom_ptr].
    # Condition: A[top_ptr] <= A[bottom_ptr] / 2
    # Using integer arithmetic: A[top_ptr] * 2 <= A[bottom_ptr]
    if A[top_ptr] * 2 <= A[bottom_ptr]:
        # Found a valid pair (A[top_ptr] as top, A[bottom_ptr] as bottom).
        # We use these two mochi to form one kagamimochi.
        count += 1
        # Move to the next potential top mochi. A[top_ptr] is now used.
        top_ptr += 1
        # Move to the next potential bottom mochi. A[bottom_ptr] is now used.
        bottom_ptr += 1
    else:
        # A[bottom_ptr] is too small to be a bottom for A[top_ptr].
        # Since the array A is sorted in ascending order, A[top_ptr] also cannot be placed
        # on any mochi with an index less than bottom_ptr (because they are <= A[bottom_ptr]).
        # Therefore, A[top_ptr] requires a larger mochi to be its bottom.
        # Advance bottom_ptr to search for a larger potential bottom for the current A[top_ptr].
        bottom_ptr += 1

# Print the maximum number of kagamimochi that can be made.
print(count)