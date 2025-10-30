import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Convert to 0-based indexing for internal use
    a = A

    # Precomputation
    pos = defaultdict(list)
    for i in range(N):
        pos[a[i]].append(i)

    next_idx = [N] * N
    last_pos = {}
    for i in range(N - 1, -1, -1):
        if a[i] in last_pos:
            next_idx[i] = last_pos[a[i]]
        last_pos[a[i]] = i

    def count_neq(start, end, value):
        # Count elements k such that start < k < end and a[k] != value
        # Range is exclusive (start, end). Indices start + 1 to end - 1.
        
        total_in_range = end - start - 1
        if total_in_range <= 0: # Check if range is empty or single element
            return 0
        
        # Count occurrences of value in a[start + 1 ... end - 1]
        # Indices in pos[value] are sorted.
        # We need indices k such that start < k < end
        # Use bisect_left to find count of indices >= start + 1 and < end
        
        value_indices = pos.get(value, []) # Handle case where value is not in pos (e.g., for non-existent values)
        
        # Number of indices < start + 1
        l = bisect_left(value_indices, start + 1)
        # Number of indices < end
        r = bisect_left(value_indices, end)
        
        count_val_in_range = r - l
        
        return total_in_range - count_val_in_range

    # DP
    dp = [0] * (N + 1)

    # dp[i] = minimum operations to clear a[i...N-1]

    for i in range(N - 1, -1, -1):
        # Option 1: Handle a[i] and then clear a[i+1...N-1].
        # This corresponds to scenarios where a[i] is removed and a[i+1] becomes the new first element.
        # The cost includes handling a[i] and minimum operations for the rest.
        # The minimum cost to process a[i] alone seems to be implicitly 1 here,
        # possibly by deleting a block of size 1 starting at a[i].
        dp[i] = 1 + dp[i+1]

        # Option 2: Handle a[i] together with a later occurrence a[j] of the same value.
        # Find the next occurrence of a[i] after index i.
        j = next_idx[i]

        if j < N:
            # To delete a[i] and a[j] together (as part of the same block),
            # all elements a[k] between i and j (i < k < j) not equal to a[i]
            # must be moved out of the way. This requires count_neq(i, j, a[i]) swaps.
            c = count_neq(i, j, a[i])
            
            # The total cost is swaps + 1 delete (for the block containing a[i] and a[j])
            # + minimum operations to clear the remaining sequence.
            # The remaining sequence is effectively a[j+1...N-1].
            # This recurrence implies the intermediate non-a[i] elements are handled
            # as part of the swaps or subsequent steps without extra cost beyond dp[j+1].
            dp[i] = min(dp[i], c + 1 + dp[j + 1])

    print(dp[0])


T = int(sys.stdin.readline())
for _ in range(T):
    solve()