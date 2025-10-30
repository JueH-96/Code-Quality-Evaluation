# YOUR CODE HERE
import sys

# Read input using fast I/O
def get_ints():
    # Helper function to read a line of space-separated integers
    return map(int, sys.stdin.readline().strip().split())

def solve():
    # Read N (number of product types) and M (maximum budget)
    N, M = get_ints()
    # Read the list of prices P_i for each product type
    P = list(get_ints())

    # Function to calculate total units (K) and total cost (C) for a given marginal cost threshold lambda_val.
    # It determines the number of units k_i for each product i such that the marginal cost of the k_i-th unit is <= lambda_val,
    # and the marginal cost of the (k_i+1)-th unit is > lambda_val.
    # It also returns the list of calculated k_i values.
    def check(lambda_val):
        total_k = 0 # Total number of units bought
        total_cost = 0 # Total cost incurred
        k_vals = [] # List to store the number of units k_i bought for each product i
        
        for i in range(N):
            # Calculate k_i for product i based on lambda_val.
            # The cost to buy k units of product i is k^2 * P[i].
            # The marginal cost of buying the k-th unit (given k-1 units already bought) is 
            # k^2 * P[i] - (k-1)^2 * P[i] = (k^2 - (k^2 - 2k + 1)) * P[i] = (2k - 1) * P[i].
            
            # We want to find the maximum integer k >= 0 such that the marginal cost of the k-th unit is <= lambda_val.
            # For k=0, the concept of marginal cost isn't well-defined, but we buy 0 units.
            # For k=1, the marginal cost is (2*1 - 1) * P[i] = P[i].
            # If lambda_val < P[i], we cannot even afford the first unit based on this threshold, so k_i = 0.
            
            if lambda_val < P[i]:
                 k_i = 0
            else:
                 # If lambda_val >= P[i], we should buy at least one unit.
                 # We are looking for the maximum integer k >= 1 such that (2k - 1) * P[i] <= lambda_val.
                 # Rearranging the inequality:
                 # 2k - 1 <= lambda_val / P[i]
                 # 2k <= lambda_val / P[i] + 1
                 # k <= (lambda_val / P[i] + 1) / 2
                 # Since k must be an integer, the maximum k is floor( (lambda_val / P[i] + 1) / 2 ).
                 
                 # Using integer arithmetic for precision and efficiency:
                 # k_i = floor( (lambda_val // P[i] + lambda_val % P[i] / P[i] + 1) / 2 )
                 # A simpler integer arithmetic version is k_i = (lambda_val // P[i] + 1) // 2.
                 # Let's verify this formula:
                 # Example P_i=4. 
                 # lambda=3: k_i=(3//4+1)//2 = (0+1)//2=0. Correct.
                 # lambda=4: k_i=(4//4+1)//2 = (1+1)//2=1. Correct. MC(k=1)=4.
                 # lambda=11: k_i=(11//4+1)//2 = (2+1)//2=1. Correct. MC(k=2)=12 > 11.
                 # lambda=12: k_i=(12//4+1)//2 = (3+1)//2=2. Correct. MC(k=2)=12 <= 12.
                 # The formula works correctly for determining k_i based on the marginal cost threshold.
                 k_i = (lambda_val // P[i] + 1) // 2
            
            # Store the calculated k_i
            k_vals.append(k_i)
            
            # Calculate the cost for k_i units of product i and add to totals.
            # Python's built-in integers handle arbitrary size, preventing overflow.
            term_cost = k_i * k_i * P[i]
            total_k += k_i
            total_cost += term_cost
            
        return total_k, total_cost, k_vals

    # Binary search for the largest lambda (marginal cost threshold) 
    # such that the total cost of items bought based on this threshold does not exceed M.
    low = 0 # Minimum possible value for lambda (technically -1 could work, but 0 is fine)
    
    # Determine a safe upper bound for lambda.
    # The maximum marginal cost could occur for the last affordable item.
    # If k units are bought, cost is k^2 P_i <= M. So k <= sqrt(M/P_i).
    # The marginal cost is (2k-1)P_i approx 2k P_i = 2 * sqrt(M/P_i) * P_i = 2 * sqrt(M*P_i).
    # Max value of M is 10^18, max P_i is 2*10^9.
    # Max marginal cost is approx 2 * sqrt(10^18 * 2*10^9) = 2 * sqrt(2 * 10^27) approx 9 * 10^13.
    # A safe upper bound could be 10^14. Even safer: 2 * M + 2 ensures any possible cost is covered.
    high = 2 * 10**18 + 2 
    
    optimal_lambda = 0 # Initialize optimal_lambda found by binary search

    # Binary search loop: searches for lambda in the range [low, high)
    # It finds the largest `low` such that `check(low)` results in cost <= M.
    while high - low > 1:
        mid = (low + high) // 2
        
        # Check the feasibility of the midpoint lambda value
        _k, current_cost, _k_vals_mid = check(mid)

        if current_cost <= M:
            # If cost is within budget, this lambda is feasible. Try potentially higher lambda values.
            low = mid
        else:
            # If cost exceeds budget, this lambda is too high. Reduce the upper bound.
            high = mid

    # After the loop, 'low' is the largest lambda such that the total cost is <= M.
    optimal_lambda = low
    
    # Calculate the final total units (final_k) and total cost (final_cost)
    # using the determined optimal lambda. Also get the list of units per product (k_values).
    final_k, final_cost, k_values = check(optimal_lambda)
    
    # Calculate the remaining budget after buying items based on the optimal lambda threshold.
    remaining_budget = M - final_cost
    
    # Now, consider buying additional items greedily using the remaining budget.
    # The candidates are the (k_i+1)-th item for each product type i.
    # Calculate the marginal cost for each of these potential next items.
    additional_items_costs = []
    for i in range(N):
        k_i = k_values[i]
        # Marginal cost of the (k_i+1)-th unit is (2*(k_i+1) - 1) * P[i] = (2*k_i + 1) * P[i].
        marginal_cost_next = (2 * k_i + 1) * P[i]
        additional_items_costs.append(marginal_cost_next)

    # Sort the marginal costs of these potential additional items in ascending order.
    additional_items_costs.sort()
    
    # Greedily add items one by one, starting from the cheapest marginal cost,
    # as long as the remaining budget allows.
    additional_count = 0
    for cost in additional_items_costs:
        # Check if the current cheapest additional item can be afforded.
        # The constraint P_i >= 1 ensures cost >= 1.
        if remaining_budget >= cost:
             # If yes, buy it: decrease budget and increment additional items count.
             remaining_budget -= cost
             additional_count += 1
        else:
             # If not, we cannot afford this item or any subsequent items (since they are sorted by cost).
             # Stop the greedy process.
             break 

    # The final answer is the sum of items bought based on the lambda threshold 
    # plus the items bought greedily with the remaining budget.
    print(final_k + additional_count)

# Execute the main function to solve the problem
solve()