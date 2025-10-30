# YOUR CODE HERE
import sys

def solve():
    # Read N (total number of items) and M (number of items to select) from input
    N, M = map(int, sys.stdin.readline().split())

    # Initialize lists to store values/capacities for each item type
    pull_tabs = []     # List of happiness values for type 0 items (pull-tab cans)
    regular_cans = []  # List of happiness values for type 1 items (regular cans)
    can_openers = []   # List of capacities (X_i) for type 2 items (can openers)

    # Read N items and categorize them based on their type T_i
    for _ in range(N):
        T, X = map(int, sys.stdin.readline().split())
        if T == 0:
            pull_tabs.append(X)
        elif T == 1:
            regular_cans.append(X)
        else: # T == 2
            can_openers.append(X)

    # Sort items of each type by their associated value (happiness for cans, capacity for openers)
    # in descending order. This allows us to greedily pick the best items of each type.
    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)

    # Get the count of items of each type
    N0 = len(pull_tabs)
    N1 = len(regular_cans)
    N2 = len(can_openers)

    # Compute prefix sums for efficient calculation of total happiness/capacity for top items.
    # P_prefix_sum[p] stores the sum of happiness of the top p pull-tab cans.
    # Index 0 is 0, P_prefix_sum[1] is happiness of the best pull-tab can, etc.
    P_prefix_sum = [0] * (N0 + 1)
    for i in range(N0):
        P_prefix_sum[i+1] = P_prefix_sum[i] + pull_tabs[i]

    # R_prefix_sum[k] stores the sum of happiness of the top k regular cans.
    R_prefix_sum = [0] * (N1 + 1)
    for i in range(N1):
        R_prefix_sum[i+1] = R_prefix_sum[i] + regular_cans[i]

    # C_prefix_sum[o] stores the sum of capacities of the top o can openers.
    C_prefix_sum = [0] * (N2 + 1)
    for i in range(N2):
        C_prefix_sum[i+1] = C_prefix_sum[i] + can_openers[i]

    # Initialize the maximum happiness found so far to 0.
    max_happiness = 0
    
    # Initialize o_count, the minimum number of can openers needed for capacity k.
    # This variable acts as a pointer in the can_openers prefix sum array, 
    # efficiently updated using a two-pointer technique as k increases.
    o_count = 0 

    # Iterate through `k`, the number of regular cans selected (and intended to be opened).
    # We assume we select the top `k` regular cans based on their happiness value.
    # The loop iterates from k=0 up to min(M, N1).
    # k cannot exceed M (total items selected) and cannot exceed N1 (total regular cans available).
    for k in range(min(M, N1) + 1):
        
        # Find the minimum number of can openers `o_count` required to achieve total capacity `k`.
        # Since `k` increases monotonically, the required `o_count` is non-decreasing.
        # We update `o_count` incrementally from its previous value.
        # `o_count` represents the *number* of openers taken.
        # C_prefix_sum[o_count] is the capacity provided by the top `o_count` openers.
        while o_count < N2 and C_prefix_sum[o_count] < k:
            o_count += 1
        
        # After the loop, check if the required capacity `k` is actually met.
        # This check is crucial if `o_count` reached `N2` (all openers used).
        if C_prefix_sum[o_count] < k:
             # If the total capacity of the top `o_count` openers (potentially all N2 openers) 
             # is still less than `k`, then it's impossible to open `k` cans.
             # Since `k` will only increase in subsequent iterations, we can break the loop early.
             break

        # Calculate the total number of items needed for k regular cans and o_count can openers.
        items_needed = k + o_count

        # Check if the number of items needed exceeds the total allowed items M.
        if items_needed > M:
            # If `items_needed` exceeds M, this configuration for `k` is not possible.
            # We continue to the next `k`. It's possible a larger `k` could be feasible if
            # `o_count` doesn't increase, keeping `k + o_count` within the limit M.
            continue

        # Calculate the required number of pull-tab cans `p` to reach exactly M items.
        p = M - items_needed
        
        # Check if we have enough pull-tab cans available. We need the top `p` pull-tab cans.
        # The check p >= 0 is implicitly handled by `items_needed <= M`.
        if p > N0:
            # If we need `p` pull-tab cans but only `N0` are available, this `k` is invalid.
            continue
        
        # If all checks pass, this combination (k regular cans, o_count openers, p pull-tabs) is valid.
        # Calculate the total happiness for this combination using prefix sums.
        current_happiness = R_prefix_sum[k] + P_prefix_sum[p]
        
        # Update the overall maximum happiness found so far.
        max_happiness = max(max_happiness, current_happiness)

    # Print the final maximum happiness calculated over all valid configurations.
    print(max_happiness)

# Execute the solve function to run the program logic
solve()