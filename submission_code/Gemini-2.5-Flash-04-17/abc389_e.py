import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))

    # Binary search for the maximum marginal cost threshold T (exclusive).
    # We check if buying all units with marginal cost strictly less than T is affordable within budget M.
    # The possible marginal costs are (2k + 1) * P_i for k >= 0, i=1...N.
    # The minimum possible marginal cost is min(P_i) >= 1.
    # A safe upper bound for the threshold T.
    # If we buy k units of product i, cost is k^2 * P_i. If k^2 * P_i <= M, k <= sqrt(M/P_i).
    # The marginal cost of the k-th unit is (2k - 1) * P_i.
    # If k is around sqrt(M/P_i), the marginal cost is around (2*sqrt(M/P_i) - 1) * P_i which is roughly 2*sqrt(M*P_i).
    # Max M = 10^18, max P_i = 2 * 10^9. Max marginal cost is approx 2*sqrt(10^18 * 2*10^9) = 2*sqrt(2*10^27) approx 2 * 1.414e13.5 approx 8.9e13.
    # A threshold slightly larger than the maximum possible relevant marginal cost should be the upper bound.
    # A generous upper bound like 4e14 + max(P_i) is safe. max(P_i) is up to 2e9.
    # 4e14 + 2e9 is sufficient. Let's use 400002000000000.

    low = 0
    high = 400002000000000 # A safe upper bound for marginal cost threshold (exclusive)
    ans_mc_threshold = 0

    # Binary search for the largest integer threshold `ans_mc_threshold`
    # such that the cost of buying all units with marginal cost strictly less than `ans_mc_threshold` is <= M.
    while low <= high:
        mid = (low + high) // 2

        current_cost = 0
        
        # Calculate the cost of buying all units with marginal cost strictly less than mid.
        # The number of units k_i of product i with marginal cost strictly less than mid is
        # the largest k_i >= 0 such that (2*k_i - 1)*p_i < mid.
        # Using integer arithmetic for mid >= 0 and p_i >= 1:
        # 2*k_i - 1 < mid / p_i
        # 2*k_i - 1 <= (mid - 1) // p_i  (for mid >= 1)
        # 2*k_i <= (mid - 1) // p_i + 1
        # k_i <= ((mid - 1) // p_i + 1) // 2
        # So k_i = ((mid - 1) // p_i + 1) // 2 for mid >= 1.
        # For mid = 0, k_i must be 0. The formula ((0-1)//p_i + 1)//2 = (-1//p_i + 1)//2 = 0 for p_i >= 1.
        # The formula ((mid - 1) // p_i + 1) // 2 works for mid >= 0 and p_i >= 1.
        
        for p_i in P:
            # Formula for k_i such that (2*k_i - 1)*p_i < mid.
            # This formula works for mid >= 0 and p_i >= 1.
            k_i = ((mid - 1) // p_i + 1) // 2
            
            # Cost for product i with k_i units is k_i^2 * p_i
            cost_i = k_i * k_i * p_i
            
            # Check if adding cost_i exceeds M before adding.
            # This is important because sum of costs might exceed M but individual cost_i might not be M+1.
            # Python handles large integers, so simple addition is fine, but checking against M early
            # can prune the loop over products.
            if current_cost + cost_i > M:
                 current_cost = M + 1 # Mark as exceeding budget
                 break # Optimization: Exit loop over products

            current_cost += cost_i # Safe to add if check passed or if no overflow concern

        if current_cost <= M:
            # All units with marginal cost < mid are affordable.
            # The optimal threshold could be mid or higher.
            ans_mc_threshold = mid
            low = mid + 1
        else:
            # Threshold mid is too high, not all units with mc < mid are affordable.
            # Need a lower threshold.
            high = mid - 1

    # Now calculate the total units and base cost for the determined threshold ans_mc_threshold.
    # ans_mc_threshold is the maximum T such that cost(<T) <= M.
    total_units_base = 0
    total_cost_base = 0
    next_marginal_costs = [] # Store marginal costs of the next available unit for each product

    for p_i in P:
        # Number of units k_i of product i with marginal cost strictly less than ans_mc_threshold.
        # Using the simplified formula (correct for >= 0 threshold).
        k_i = ((ans_mc_threshold - 1) // p_i + 1) // 2 if ans_mc_threshold > 0 else 0
        
        total_units_base += k_i
        total_cost_base += k_i * k_i * p_i

        # The marginal cost of the next unit (the (k_i + 1)-th unit) is (2*k_i + 1) * p_i.
        # This is a candidate for additional purchase if affordable.
        next_marginal_costs.append((2 * k_i + 1) * p_i)

    # Sort the marginal costs of the next available units.
    next_marginal_costs.sort()

    # Buy additional units greedily using the remaining budget.
    remaining_budget = M - total_cost_base
    additional_units = 0

    for next_mc in next_marginal_costs:
        # Marginal cost must be positive since P_i >= 1 and (2*k_i + 1) >= 1.
        if remaining_budget >= next_mc:
            # Can afford this unit.
            remaining_budget -= next_mc
            additional_units += 1
        else:
            # Cannot afford this unit or any subsequent unit (since costs are sorted).
            break

    print(total_units_base + additional_units)

solve()