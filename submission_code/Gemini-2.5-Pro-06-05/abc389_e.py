# YOUR CODE HERE
import sys

def main():
    """
    This program solves the problem by binary searching on the marginal cost threshold.
    """
    # Read inputs from standard input
    input = sys.stdin.readline
    try:
        N, M = map(int, input().split())
        P = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle cases like empty input at the end of file in some environments.
        return

    def get_cost(x):
        """
        Calculates the total cost if we buy all units
        with a marginal cost up to the threshold `x`.
        """
        total_cost = 0
        for p_i in P:
            # Number of items `k` for product `i` s.t. marginal cost (2k-1)p_i <= x
            # This simplifies to k = (x // p_i + 1) // 2
            k = (x // p_i + 1) // 2
            
            # Total cost for product `i` is k^2 * p_i
            total_cost += k * k * p_i
            
            # Optimization: if cost already exceeds budget, no need to continue.
            if total_cost > M:
                return M + 1
        return total_cost

    # Binary search for the optimal marginal cost threshold `x_opt`.
    # `x_opt` is the largest integer `x` such that `get_cost(x)` is within budget `M`.
    low = 0
    # A safe upper bound for marginal cost. Derived from analysis:
    # Max x is roughly 2*sqrt(M*P_max) ~ 2*sqrt(10^18 * 2*10^9) ~ 9*10^13.
    # 2*10^14 is a safe, round number.
    high = 2 * 10**14 
    x_opt = 0

    while low <= high:
        mid = (low + high) // 2
        
        cost = get_cost(mid)
        
        if cost <= M:
            # `mid` is a feasible threshold. Store it and try a higher `x`.
            x_opt = mid
            low = mid + 1
        else:
            # `mid` is too high. Need to lower the threshold.
            high = mid - 1
            
    # With the optimal threshold `x_opt`, calculate the base items and cost.
    items_base = 0
    cost_base = 0
    for p_i in P:
        k = (x_opt // p_i + 1) // 2
        items_base += k
        cost_base += k * k * p_i
    
    # We have `M - cost_base` yen remaining.
    # The next cheapest items all have a marginal cost of `x_opt + 1`.
    # We can buy `(M - cost_base) // (x_opt + 1)` more items.
    rem_budget = M - cost_base
    
    # Since x_opt >= 0, x_opt + 1 >= 1. No division by zero.
    additional_items = rem_budget // (x_opt + 1)
    
    total_items = items_base + additional_items
    
    print(total_items)

if __name__ == "__main__":
    main()