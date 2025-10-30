import sys
import heapq
from collections import defaultdict

def solve():
    # Read N (length of array A) and Q (number of queries)
    N, Q = map(int, sys.stdin.readline().split())

    # Read the initial array A. Store it as a list for mutable access.
    A = list(map(int, sys.stdin.readline().split()))

    # counts[x] will store the frequency of number x in the current array A.
    # defaultdict(int) conveniently initializes the count to 0 for any new key.
    counts = defaultdict(int)
    for x in A:
        counts[x] += 1

    # H is a min-heap that stores all non-negative integers k such that:
    # 1. 0 <= k <= N (because the MEX of an array of N elements is always <= N).
    # 2. k is currently NOT present in the array A (i.e., counts[k] == 0).
    # The smallest element at the top of this heap (H[0]) will be the current MEX.
    H = []
    for k in range(N + 1): # Iterate from 0 up to N to find initial missing numbers
        if counts[k] == 0:
            heapq.heappush(H, k)

    # Process each query in the given order
    for _ in range(Q):
        # Read the index i (1-based) and the new value x for the current query
        i, x = map(int, sys.stdin.readline().split())
        idx = i - 1 # Convert 1-based index to 0-based array index

        old_val = A[idx] # Get the value that is currently at A[idx]
        A[idx] = x       # Update the array at index idx with the new value x

        # --- Step 1: Adjust frequency count for old_val ---
        counts[old_val] -= 1
        # If the count of old_val drops to 0, it means it is no longer present in A.
        # If old_val is a potential MEX candidate (i.e., 0 <= old_val <= N),
        # we add it to our min-heap of missing numbers.
        if counts[old_val] == 0:
            if old_val <= N: # Only values <= N can be MEX candidates
                heapq.heappush(H, old_val)

        # --- Step 2: Adjust frequency count for new_val (x) ---
        counts[x] += 1
        # If the count of new_val just became 1 (meaning it was 0 and is now present),
        # and new_val is a potential MEX candidate (i.e., 0 <= new_val <= N),
        # it means new_val was potentially in our min-heap H.
        # We don't remove it explicitly from H here because heapq does not support
        # efficient arbitrary element removal. Instead, we use "lazy deletion" strategy
        # which is handled in the next step.
        # If x > N, it would never have been in H, so no special handling needed here.

        # --- Step 3: Find and print the current MEX ---
        # The smallest element at the top of the heap (H[0]) is the current candidate for MEX.
        # Due to lazy deletion, it's possible that H[0] is a number that is now present in A
        # (its count is no longer 0). We repeatedly pop from H until we find a number
        # whose count is genuinely 0.
        while counts[H[0]] > 0:
            heapq.heappop(H)
        
        # After the loop, H[0] is guaranteed to be the smallest non-negative integer
        # (0 <= k <= N) that is not present in A. This is the MEX.
        print(H[0])

# Call the solve function to run the program
solve()