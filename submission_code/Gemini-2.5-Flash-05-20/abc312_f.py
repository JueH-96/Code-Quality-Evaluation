import sys
import bisect

def solve():
    N, M = map(int, sys.stdin.readline().split())
    items = []
    for _ in range(N):
        items.append(list(map(int, sys.stdin.readline().split())))

    pull_tabs = sorted([x for t, x in items if t == 0], reverse=True)
    regular_cans = sorted([x for t, x in items if t == 1], reverse=True)
    can_openers = sorted([x for t, x in items if t == 2], reverse=True)

    # Precompute prefix sums for happiness values of pull-tabs and regular cans
    # pull_tabs_prefix_sum[i] stores the sum of happiness of the top i pull-tabs
    pull_tabs_prefix_sum = [0] * (len(pull_tabs) + 1)
    for i in range(len(pull_tabs)):
        pull_tabs_prefix_sum[i+1] = pull_tabs_prefix_sum[i] + pull_tabs[i]

    # regular_cans_prefix_sum[i] stores the sum of happiness of the top i regular cans
    regular_cans_prefix_sum = [0] * (len(regular_cans) + 1)
    for i in range(len(regular_cans)):
        regular_cans_prefix_sum[i+1] = regular_cans_prefix_sum[i] + regular_cans[i]
    
    # Precompute prefix sums for capacities of can openers
    # can_openers_prefix_sum[i] stores the sum of capacities of the top i can openers
    can_openers_prefix_sum = [0] * (len(can_openers) + 1)
    for i in range(len(can_openers)):
        can_openers_prefix_sum[i+1] = can_openers_prefix_sum[i] + can_openers[i]

    max_total_happiness = 0

    # Iterate over the number of regular cans (`num_r_cans`) we choose to open.
    # `num_r_cans` can range from 0 up to `M` (total items allowed) or `len(regular_cans)`
    for num_r_cans in range(min(M + 1, len(regular_cans) + 1)):
        current_happiness_from_r = regular_cans_prefix_sum[num_r_cans]
        
        # Determine the minimum number of openers required to open `num_r_cans`.
        # We need at least `num_r_cans` total capacity.
        # `bisect_left(a, x)` returns an insertion point which comes before (to the left of)
        # any existing entries of `x` in `a`. Here, `a` is `can_openers_prefix_sum`,
        # and `x` is `num_r_cans`. The returned index will be the minimum `k` such that
        # `can_openers_prefix_sum[k] >= num_r_cans`.
        num_openers = bisect.bisect_left(can_openers_prefix_sum, num_r_cans)
        
        # If the required number of openers exceeds available openers, this combination is not possible.
        # This occurs if bisect_left returns an index greater than len(can_openers).
        if num_openers > len(can_openers):
            continue 
        
        # Calculate total items chosen so far (regular cans + openers)
        items_chosen_so_far = num_r_cans + num_openers
        
        # If we have chosen more items than allowed by M, this combination is not feasible.
        if items_chosen_so_far > M:
            continue
        
        # Calculate remaining slots for pull-tabs
        remaining_slots_for_pull_tabs = M - items_chosen_so_far
        
        # Add happiness from the best pull-tabs that fit into remaining slots
        happiness_from_pull_tabs = pull_tabs_prefix_sum[min(remaining_slots_for_pull_tabs, len(pull_tabs))]
        
        # Calculate total happiness for this specific combination of (num_r_cans, num_openers, num_p_cans)
        current_total_happiness = current_happiness_from_r + happiness_from_pull_tabs
        
        # Update overall maximum happiness
        max_total_happiness = max(max_total_happiness, current_total_happiness)

    print(max_total_happiness)