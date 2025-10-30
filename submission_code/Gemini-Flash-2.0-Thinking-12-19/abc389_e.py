import sys

def solve():
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    P = list(map(int, sys.stdin.readline().split()))

    # Function to calculate total units and total cost for a given marginal cost threshold C
    # We buy k_i units of product i if the marginal cost of the k_i-th unit (2*k_i - 1) * P_i <= C for k_i >= 1.
    # The number of units k_i is the largest integer such that (2*k_i - 1)*P_i <= C (for k_i >= 1).
    # This gives k_i = floor((C / P_i + 1) / 2). Using integer division: k_i = (C // P_i + 1) // 2.
    # This formula also correctly gives k_i = 0 when P_i > C.
    def check(C):
        total_cost = 0
        
        for p in P:
            if p == 0: continue # Should not happen based on constraints P_i >= 1

            # Calculate k_i based on threshold C
            k_i = (C // p + 1) // 2
            
            # Calculate cost for k_i units: k_i^2 * p
            # Python handles large integers, so k_i * k_i * p is fine.
            cost_i = k_i * k_i * p 
            
            # Check if adding cost_i would exceed the remaining budget M - total_cost
            # This check prevents total_cost from exceeding M and correctly identifies if C is too high.
            # M - total_cost can be large (up to M). The check is safe with arbitrary precision integers.
            if total_cost > M - cost_i:
                 return False # Total cost will exceed M

            total_cost += cost_i
        
        # If we iterated through all products without exceeding M, then this C is affordable.
        return True

    # Binary search for the maximum threshold C.
    # The threshold C represents the maximum allowed marginal cost for the last unit bought.
    # The optimal number of units is achieved by buying units with the lowest marginal costs
    # up to some effective threshold C_opt, plus possibly some additional units at the next marginal cost.
    # The range for C: [0, MAX_C].
    # The maximum relevant marginal cost is estimated around 10^14 (derived from M=10^18, P_i>=1).
    # Using 2e14 + a margin as the upper bound should be sufficient.
    low = 0
    high = 2 * 10**14 + 200 # A sufficiently large upper bound for C

    # We are searching for the maximum C in the range [low, high] such that check(C) is True.
    # Standard binary search template for finding the last True value.
    ans_C = 0 # Stores the maximum C for which check(C) is True
    
    L = low
    R = high 
    
    while L <= R:
        mid = L + (R - L) // 2
        if check(mid):
            ans_C = mid # mid is a possible answer, try a larger threshold
            L = mid + 1
        else:
            R = mid - 1 # mid is too high, try a smaller threshold

    # After binary search, C_opt is the maximum threshold C such that Cost(C) <= M.
    C_opt = ans_C

    # Calculate the total units K_opt and total cost Cost_opt for the threshold C_opt.
    # These calculations are guaranteed not to exceed M for Cost_opt based on the binary search result.
    K_opt = 0
    Cost_opt = 0 
    
    for p in P:
        k_i = (C_opt // p + 1) // 2
        K_opt += k_i
        Cost_opt += k_i * k_i * p # Python handles large integers

    # Find the minimum marginal cost strictly greater than C_opt.
    # This is the cost of the next cheapest unit available after buying all units with marginal cost <= C_opt.
    # Iterate through all products i, calculate k_i for C_opt, and find the marginal cost of the (k_i+1)-th unit, which is (2*k_i + 1) * P_i.
    # Find the minimum of these values that is strictly greater than C_opt.
    
    # Initialize C_next to a value larger than any possible relevant marginal cost.
    # Max marginal cost is estimated around 10^14. Use a value larger than the binary search upper bound for C.
    C_next = 3 * 10**14 + 300 

    found_C_next = False
    for p in P:
        # k_i is the number of units of product p bought with threshold C_opt
        k_i = (C_opt // p + 1) // 2
        
        # The marginal cost of the next unit (k_i + 1)-th unit
        next_marginal_cost = (2 * k_i + 1) * p
        
        # Check if this is the smallest marginal cost strictly greater than C_opt found so far.
        if next_marginal_cost > C_opt:
            if not found_C_next or next_marginal_cost < C_next:
                 C_next = next_marginal_cost
                 found_C_next = True

    # Calculate the remaining budget after buying units up to threshold C_opt.
    M_rem = M - Cost_opt
    
    # Calculate additional units that can be bought with the remaining budget.
    # These additional units must have the minimum available marginal cost, which is C_next.
    k_extra = 0
    if found_C_next and M_rem > 0:
        # We can buy floor(M_rem / C_next) units, each costing C_next.
        # Integer division handles the floor automatically.
        k_extra = M_rem // C_next

    # The total number of units is the units bought with threshold C_opt plus the additional units.
    total_units = K_opt + k_extra

    print(total_units)

solve()