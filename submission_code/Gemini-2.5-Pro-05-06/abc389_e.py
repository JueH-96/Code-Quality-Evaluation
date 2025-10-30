import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))

    # Calculates k, the number of items of a product (with cost coefficient P_val)
    # such that the marginal cost of the k-th item is at most cost_limit.
    # Marginal cost of k-th item: (2*(k-1)+1)*P_val = (2k-1)*P_val
    def count_items_le_marginal_cost(cost_limit, P_val):
        # P_val is guaranteed to be >= 1 by constraints.
        
        if cost_limit < P_val: # Cost of 1st item is P_val. If limit is less, cannot buy any.
            return 0
        
        # We need to find max k >= 1 such that (2k-1) * P_val <= cost_limit
        # Let val_for_2k_minus_1 = floor(cost_limit / P_val) using integer division
        val_for_2k_minus_1 = cost_limit // P_val
        
        # Now we have 2k-1 <= val_for_2k_minus_1
        # 2k <= val_for_2k_minus_1 + 1
        # k = floor((val_for_2k_minus_1 + 1) / 2)
        k = (val_for_2k_minus_1 + 1) // 2
        return k

    low = 0
    # Max marginal cost can be around 10^14. A safe upper bound is used.
    # Minimum non-zero marginal cost is 1 (if some P_i = 1).
    high = 2 * (10**14) + 7 

    ans_L = 0 # Stores the optimal lambda value found
    ans_N_at_L = 0 # Number of items if marginal cost <= ans_L
    ans_C_at_L = 0 # Cost for these items

    while low <= high:
        cur_L = low + (high - low) // 2
        # cur_L is the candidate for max_permissible_marginal_cost

        current_num_items = 0
        current_total_cost = 0
        
        for p_val in P:
            k = count_items_le_marginal_cost(cur_L, p_val)
            current_num_items += k
            
            cost_for_this_product = k * k * p_val
            current_total_cost += cost_for_this_product
            
            if current_total_cost > M: # Optimization: early exit
                break
        
        if current_total_cost <= M:
            ans_L = cur_L
            ans_N_at_L = current_num_items
            ans_C_at_L = current_total_cost
            low = cur_L + 1
        else: # current_total_cost > M
            high = cur_L - 1
            
    remaining_budget = M - ans_C_at_L
    total_items_bought = ans_N_at_L
    
    cost_of_next_tier_item = ans_L + 1
    
    # Since ans_L >= 0, cost_of_next_tier_item is >= 1.
    # No division by zero risk for remaining_budget // cost_of_next_tier_item.
    
    num_potential_next_tier_items = 0
    for p_val in P:
        k_already_bought = count_items_le_marginal_cost(ans_L, p_val)
        mc_of_next = (2 * k_already_bought + 1) * p_val
        
        if mc_of_next == cost_of_next_tier_item:
            num_potential_next_tier_items += 1

    if num_potential_next_tier_items > 0:
        num_can_afford_next_tier = remaining_budget // cost_of_next_tier_item
        items_to_add = min(num_potential_next_tier_items, num_can_afford_next_tier)
        total_items_bought += items_to_add

    sys.stdout.write(str(total_items_bought) + "
")

solve()