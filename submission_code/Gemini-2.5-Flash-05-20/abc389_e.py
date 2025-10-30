# YOUR CODE HERE
import sys

def solve():
    # Read N and M from the first line of input
    N, M = map(int, sys.stdin.readline().split())
    # Read the list of P_i values from the second line of input
    P = list(map(int, sys.stdin.readline().split()))

    # Binary search for the maximum marginal cost threshold C.
    # C is the highest marginal cost we are willing to pay for any additional unit.
    # If we decide on a C, for each product i, we calculate how many units k_i
    # we can buy such that the marginal cost of the k_i-th unit is <= C.
    
    # Determine the search range for C:
    # The smallest possible C is 0 (meaning we buy nothing if the marginal cost is greater than 0).
    low_C = 0
    # The largest possible C:
    # If M = 10^18 and min(P_i) = 1, then k_i for a single product could be up to sqrt(10^18) = 10^9 units.
    # The marginal cost for the k_i-th unit is (2*k_i - 1) * P_i.
    # Max k_i is about 10^9, max P_i is 2 * 10^9.
    # So, the maximum possible marginal cost can be roughly 2 * (10^9) * (2 * 10^9) = 4 * 10^18.
    # We set a sufficiently large upper bound for `high_C` to ensure the optimal `C` is within range.
    high_C = 4 * 10**18 + 5 

    max_total_units_found = 0

    # Perform binary search
    while low_C <= high_C:
        mid_C = (low_C + high_C) // 2 # Current marginal cost threshold to check

        current_total_units = 0
        current_total_cost = 0

        # For each product type, calculate how many units to buy given mid_C
        for p_val in P:
            # We buy units as long as their marginal cost is less than or equal to mid_C.
            # The marginal cost of the k-th unit of a product with price P_i is (2*k - 1) * P_i.
            # We want to find the largest k (let's call it k_i) such that (2*k_i - 1) * p_val <= mid_C.
            #
            # Rearranging the inequality for k_i:
            # 2*k_i - 1 <= mid_C / p_val
            # 2*k_i <= mid_C / p_val + 1
            # k_i <= (mid_C / p_val + 1) / 2
            
            # Since k_i must be an integer, we take the floor of the right side.
            # In Python, `a // b` performs floor division for positive numbers.
            # The calculation `(mid_C // p_val + 1) // 2` correctly implements floor((mid_C / p_val + 1) / 2).
            # If mid_C is smaller than p_val (e.g., mid_C=5, p_val=10), `mid_C // p_val` is 0.
            # Then `(0 + 1) // 2` is 0, which correctly means 0 units are bought for this product.
            k_i = (mid_C // p_val + 1) // 2
            
            current_total_units += k_i
            # Calculate the cost for k_i units of this product: k_i^2 * p_val
            # Python's integers automatically handle arbitrary precision, so no overflow for large sums.
            current_total_cost += k_i * k_i * p_val

        # Check if the total cost for this set of units (determined by mid_C) is within budget M.
        if current_total_cost <= M:
            # If we can afford these units, this is a possible answer.
            # To maximize total units, we try to achieve even more units by
            # allowing a higher marginal cost threshold. So, we store the current
            # total units and search in the upper half of the C range.
            max_total_units_found = current_total_units
            low_C = mid_C + 1
        else:
            # If the current total cost exceeds the budget M,
            # the `mid_C` threshold is too high. We need to reduce the
            # marginal cost threshold. So, we search in the lower half of the C range.
            high_C = mid_C - 1
            
    # After the binary search loop, max_total_units_found will hold the maximum
    # number of units that can be bought within the budget M.
    print(max_total_units_found)

# Call the solve function to run the program
solve()