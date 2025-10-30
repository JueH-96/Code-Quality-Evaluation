import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    pull_tabs_values = [] # Type 0
    regular_cans_values = [] # Type 1
    can_openers_capacities = [] # Type 2
    
    for _ in range(N):
        T, X = map(int, sys.stdin.readline().split())
        if T == 0:
            pull_tabs_values.append(X)
        elif T == 1:
            regular_cans_values.append(X)
        else:
            can_openers_capacities.append(X)
            
    pull_tabs_values.sort(reverse=True)
    regular_cans_values.sort(reverse=True)
    can_openers_capacities.sort(reverse=True)
    
    n_p = len(pull_tabs_values)
    n_r = len(regular_cans_values)
    n_o = len(can_openers_capacities)
    
    # ps_X[k] = sum of top k items of type X values (happiness or capacity)
    # ps_X[0] = 0 by definition
    
    ps_p = [0] * (n_p + 1)
    for i in range(n_p):
        ps_p[i+1] = ps_p[i] + pull_tabs_values[i]
        
    ps_r = [0] * (n_r + 1)
    for i in range(n_r):
        ps_r[i+1] = ps_r[i] + regular_cans_values[i]
        
    ps_o = [0] * (n_o + 1) # Prefix sums of capacities
    for i in range(n_o):
        ps_o[i+1] = ps_o[i] + can_openers_capacities[i]

    max_overall_happiness = 0
    
    # current_num_openers stores the count of openers currently considered.
    # ps_o[current_num_openers] is the sum of capacities of these openers.
    # This variable is maintained across iterations of num_R to optimize finding
    # the minimum number of openers needed. As num_R increases, 
    # current_num_openers can only increase or stay the same.
    current_num_openers = 0 

    # Iterate over num_R, the number of regular cans selected.
    # We select the top num_R regular cans from the sorted list.
    # The number of regular cans (num_R) can range from 0 up to n_r.
    # Also, num_R cannot exceed M (total items allowed), this is implicitly handled.
    for num_R in range(min(M, n_r) + 1):
        
        happiness_from_R_cans = ps_r[num_R]
        
        num_O_final = 0 # This will be the count of openers we decide to take.
        
        if num_R == 0:
            # No regular cans selected, so no openers needed.
            num_O_final = 0
        else:
            # We need to find the smallest k_o such that ps_o[k_o] >= num_R.
            # current_num_openers is already a candidate for k_o (it was sufficient 
            # for a previous, smaller num_R, or is 0 if this is the first num_R > 0).
            # We advance current_num_openers if its capacity ps_o[current_num_openers] is less than num_R.
            while current_num_openers < n_o and ps_o[current_num_openers] < num_R:
                current_num_openers += 1
            
            # After the loop, current_num_openers is either n_o (all openers considered)
            # or ps_o[current_num_openers] >= num_R (found enough capacity with current_num_openers openers).
            
            if ps_o[current_num_openers] < num_R:
                # This condition implies that current_num_openers == n_o (all openers were taken)
                # AND their total capacity ps_o[n_o] is still less than num_R.
                # Or, if n_o == 0, then current_num_openers == 0, and ps_o[0] == 0. If num_R > 0, then 0 < num_R.
                # In this case, we cannot open num_R cans.
                # Since num_R will only increase in subsequent iterations, needing even more capacity, we can stop.
                break 
            
            num_O_final = current_num_openers
            
        items_count_so_far = num_R + num_O_final
        
        if items_count_so_far > M:
            # Not enough slots in M to take these regular cans and their openers.
            # For this num_R, it's impossible. We continue to the next num_R,
            # as a different num_R (e.g. smaller) might fit.
            continue
            
        remaining_slots = M - items_count_so_far
        
        # Fill remaining slots with pull-tab cans.
        # Take top num_P_to_take pull-tab cans based on available slots.
        num_P_to_take = min(remaining_slots, n_p)
        
        happiness_from_P_cans = ps_p[num_P_to_take]
        
        current_total_happiness = happiness_from_R_cans + happiness_from_P_cans
        
        if current_total_happiness > max_overall_happiness:
            max_overall_happiness = current_total_happiness
            
    print(max_overall_happiness)

solve()