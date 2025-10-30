import sys

def solve():
    # Read input
    N, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # dp[k] is a dictionary mapping total_sweetness (s) to minimum_total_saltiness (t)
    # using exactly k dishes. We store only Pareto optimal states for efficiency.
    # A state (s1, t1) is dominated by (s2, t2) if s2 <= s1 and t2 <= t1, and (s1, t1) != (s2, t2).
    dp = [{} for _ in range(N + 1)]
    dp[0] = {0: 0} # 0 dishes, 0 sweetness, 0 saltiness

    min_k_ans = N + 1 # Initialize with a value greater than N

    # Iterate through each dish
    for i in range(N):
        current_A = A[i]
        current_B = B[i]

        # Iterate backwards through the number of dishes k
        # This ensures that when calculating dp[k+1], we use states from dp[k]
        # that were formed *before* considering the current dish `i`.
        # dp[k] holds states using k dishes chosen from the first `i` dishes.
        # Adding dish `i` results in states with k+1 dishes chosen from the first `i+1` dishes.
        for k in range(i, -1, -1): # k goes from `i` (max dishes used before current) down to 0
            if not dp[k]: continue # Skip if no reachable states with k dishes

            # Create new states by adding the current dish (current_A, current_B) to each state in dp[k]
            new_states = []
            for s, t in dp[k].items():
                new_s = s + current_A
                new_t = t + current_B
                new_states.append((new_s, new_t))

                # Check if this new state causes a stop at k+1 dishes
                # If total sweetness > X or total saltiness > Y, we stop.
                if new_s > X or new_t > Y:
                    min_k_ans = min(min_k_ans, k + 1)

            # Merge existing states in dp[k+1] with new_states generated from dp[k]
            # Use a temporary dictionary to efficiently merge and find minimum saltiness for each sweetness.
            # k+1 is the new number of dishes.
            merged_dict = dp[k + 1].copy() # Start with existing states for k+1 dishes (using dishes from first i)

            for s, t in new_states:
                 # If sweetness s is not in merged_dict or the new saltiness t is smaller
                 if s not in merged_dict or t < merged_dict[s]:
                     merged_dict[s] = t

            # Convert dictionary back to a sorted list and rebuild the Pareto front
            # Sorting by sweetness 's' is the first step for Pareto front construction
            merged_list = sorted(merged_dict.items())

            new_dp_k_plus_1 = []
            min_salt_so_far = float('inf') # Tracks the minimum saltiness encountered so far in the sorted list
            for s, t in merged_list:
                # Add state (s, t) to the Pareto front if its saltiness 't' is strictly
                # less than the minimum saltiness found for any state with sweetness
                # less than or equal to 's' that was added previously in this list.
                # This correctly builds the Pareto front where points are sorted by increasing s and decreasing t.
                if t < min_salt_so_far:
                    new_dp_k_plus_1.append((s, t))
                    min_salt_so_far = t

            # Update dp[k+1] with the new Pareto front
            # Convert list back to dictionary for the next iteration's merging step
            # This keeps the number of states manageable by only storing non-dominated (s, t) pairs for each k.
            dp[k + 1] = dict(new_dp_k_plus_1)

    # The problem guarantees that he will stop eating eventually.
    # This means min_k_ans will be updated to a value <= N.
    print(min_k_ans)

solve()