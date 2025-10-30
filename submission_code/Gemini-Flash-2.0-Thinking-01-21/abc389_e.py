import sys

def solve():
    # Read input
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))

    # Function to calculate total units and cost for a given marginal cost threshold C
    def calculate_units_and_cost_for_threshold(C, P):
        total_units = 0
        total_cost = 0

        # Python handles large integers automatically, so no explicit big integer type is needed.

        for p in P:
            # Number of units k_i of product with price p such that the marginal cost
            # of the k-th unit (1-indexed) is less than or equal to C.
            # Marginal cost of k-th unit is (2*k - 1) * p for k >= 1.
            # We want max k_i >= 0 such that (2*k_i - 1) * p <= C.
            # If k_i=0, (2*0-1)*p = -p <= C (always true since C>=0, p>=1). This corresponds to 0 units.
            # If k_i >= 1, 2*k_i - 1 <= C / p (real division)
            # 2*k_i <= C / p + 1
            # k_i <= (C / p + 1) / 2 (real division)
            # Max integer k_i is floor((C / p + 1) / 2).
            # Using integer division for C >= 0, p >= 1: k_i = (C // p + 1) // 2
            k_i = (C // p + 1) // 2
            
            total_units += k_i
            total_cost += k_i * k_i * p
            # Optimization: If cost already exceeds M, we can stop early in the binary search check
            # This check is done in the binary search loop itself.

        return total_units, total_cost

    # Binary search for the maximum marginal cost threshold C (ans_C)
    # such that the cost of buying all units with marginal cost <= C is at most M.
    # The range of C: low = 0. High should be a value larger than any possible relevant marginal cost.
    # A relevant marginal cost (2k+1)p happens when k^2 p <= M approximately.
    # k is roughly sqrt(M/p). (2k+1)p approx 2 * sqrt(M/p) * p = 2 * sqrt(M*p).
    # Max M = 10^18, max P_i = 2*10^9. Max 2*sqrt(M*p) approx 2 * sqrt(10^18 * 2*10^9) = 2 * sqrt(2*10^27) approx 8.94 * 10^13.
    # So C can be around 10^14. A safe upper bound like 2 * 10^14 + max(P) is sufficient.
    # Let's use 2 * 10**14 + 2 * 10**9 as high.
    high = 2 * 10**14 + 2 * 10**9
    low = 0
    ans_C = 0 # This will store the largest C found so far that satisfies the condition

    # Perform integer binary search
    # We need enough iterations to narrow down the range. log2(high - low) iterations.
    # log2(2e14) approx log2(10^14) = 14 * log2(10) approx 14 * 3.32 = 46.5.
    # A few more iterations cover the range. 100 iterations is more than enough.
    
    while low <= high:
        mid = (low + high) // 2

        # Calculate cost for this threshold mid
        current_cost = 0
        for p in P:
             k_i = (mid // p + 1) // 2
             current_cost += k_i * k_i * p
             
             # Optimization: If cost already exceeds M, break early
             if current_cost > M:
                 break

        if current_cost <= M:
            # mid is a possible maximum threshold, try a larger one
            ans_C = mid
            low = mid + 1
        else:
            # mid is too high, try a smaller one
            high = mid - 1

    # After finding the maximum C (ans_C) such that cost(C) <= M
    # Calculate the base units and cost using this C_max, and also the marginal costs of the next units.
    K_base = 0
    Cost_base = 0
    next_marginal_costs = []

    for p in P:
        # Calculate k_i based on the determined threshold ans_C
        k_i = (ans_C // p + 1) // 2
        K_base += k_i
        Cost_base += k_i * k_i * p
        
        # Calculate the marginal cost of the next unit (k_i + 1)-th unit for this product.
        # This unit's marginal cost is (2*k_i + 1) * p.
        # By the definition of k_i = (ans_C // p + 1) // 2, the marginal cost of the k_i-th unit
        # (if k_i >= 1) is (2*k_i - 1)*p which is <= ans_C.
        # The marginal cost of the (k_i+1)-th unit is (2*k_i + 1)*p, which is strictly greater than ans_C
        # unless (2*k_i-1)*p == ans_C and the next potential marginal cost is also ans_C,
        # which is only possible if (2k+1)p = ans_C for two consecutive k, impossible.
        # So (2*k_i+1)*p > ans_C always holds by properties of k_i.
        next_marginal_cost = (2 * k_i + 1) * p
        
        next_marginal_costs.append(next_marginal_cost)

    # Remaining budget after buying base units
    # This should be non-negative because ans_C is the maximum C where Cost(C) <= M.
    M_rem = M - Cost_base 

    # Greedily buy additional units with the remaining budget
    # The available units we haven't bought yet are the (k_i+1)-th units for each product i,
    # with marginal costs stored in next_marginal_costs.
    # Sort these costs and buy the cheapest available ones until the budget runs out.
    next_marginal_costs.sort()

    additional_units = 0
    for marginal_cost in next_marginal_costs:
        # Check if we can afford this unit with the remaining budget
        if marginal_cost <= M_rem:
            M_rem -= marginal_cost
            additional_units += 1
        else:
            # Cannot afford this unit or any more expensive ones in the sorted list
            break

    # Total maximum units is the sum of base units and additional units
    total_units = K_base + additional_units

    print(total_units)

solve()