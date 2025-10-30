import sys
import collections

def solve():
    N = int(sys.stdin.readline())
    H_str = sys.stdin.readline().split()
    H_values = [int(x) for x in H_str]

    # According to constraints: 1 <= N <= 3000.
    # If N=0, result would be 0. If N=1, result must be 1.
    # The problem implies N >= 1, so at least one building can be chosen.
    if N == 0: # Defensive coding, not strictly needed by constraints.
        print(0)
        return
        
    max_overall_len = 1
    # If N > 0, we can always choose one building.
    
    # Group indices by height.
    # collections.defaultdict(list) automatically creates an empty list
    # for a new height key when it's first accessed.
    height_to_indices = collections.defaultdict(list)
    for i in range(N):
        height_to_indices[H_values[i]].append(i)

    # Iterate over each unique height found in the input.
    for h_val in height_to_indices:
        indices = height_to_indices[h_val]
        m = len(indices) # Number of buildings with this height.

        # Optimization: if the count of buildings with this height (m)
        # is not greater than the current max_overall_len,
        # then this height cannot possibly yield a longer arithmetic progression.
        if m <= max_overall_len:
            continue
        
        # dp[k] will be a dictionary.
        # It maps a common difference `d` to the length of an arithmetic progression
        # (AP) that:
        #   1. Consists of buildings with the current height `h_val`.
        #   2. Ends at building `indices[k]`.
        #   3. Has common difference `d`.
        # This dp table is specific to the current height h_val and its `indices`.
        # It's reset for each height.
        dp = [{} for _ in range(m)]

        # Iterate through each building `indices[k]` (with current height `h_val`)
        # as a potential end-point of an AP.
        for k in range(m):
            # Any single building `indices[k]` itself forms an AP of length 1.
            # This fact is used as the base for longer APs.
            # The max_overall_len is already initialized to at least 1.

            # Now, try to extend APs that ended at `indices[j]` (where j < k)
            # by appending `indices[k]`.
            for j in range(k):
                # `indices[j]` and `indices[k]` are two building indices.
                # Since `indices` are collected by iterating i from 0 to N-1,
                # they are already sorted: `indices[k] > indices[j]`.
                # Thus, the common difference `diff` will be positive.
                diff = indices[k] - indices[j]
                
                # Get the length of an AP ending at `indices[j]` with common difference `diff`.
                # If `dp[j]` doesn't have `diff` as a key, it means no such AP (longer than 1)
                # ending at `indices[j]` with this `diff` was found originating from an earlier building.
                # In this case, `indices[j]` is considered the start of a new AP $(indices[j], indices[k])$.
                # The length of the "progression" ending at `indices[j]` is taken as 1.
                prev_len = dp[j].get(diff, 1)
                
                # The new AP ending at `indices[k]` has length `prev_len + 1`.
                current_len = prev_len + 1
                dp[k][diff] = current_len
                
                # Update the overall maximum length found so far.
                if current_len > max_overall_len:
                    max_overall_len = current_len
    
    print(max_overall_len)

if __name__ == '__main__':
    solve()