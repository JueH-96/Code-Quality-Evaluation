# YOUR CODE HERE
import sys
import math

# Setting a higher recursion depth is generally irrelevant for iterative DP, 
# but doesn't hurt unless memory is extremely constrained.
# It's included here as a standard practice in competitive programming, though not needed for this problem.
# sys.setrecursionlimit(2000) 

def solve():
    # Read input values N (number of orders), K (max orders per shipment), X (days between shipments)
    N, K, X = map(int, sys.stdin.readline().split())
    # Read the list of placement days T for each order
    T = list(map(int, sys.stdin.readline().split()))

    # Use 1-based indexing for T for easier correspondence with problem statement indices (order 1 to N).
    # We achieve this by padding the list T with a dummy value (0) at the beginning.
    # T_1based[i] will store the placement day of order i.
    T_1based = [0] + T 
    
    # Calculate prefix sums of the placement days T.
    # P[i] stores the sum T_1 + T_2 + ... + T_i. P[0] is initialized to 0.
    # Prefix sums allow calculating the sum of T_k over any range [a, b] in O(1) time as P[b] - P[a-1].
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = P[i-1] + T_1based[i]

    # Initialize the Dynamic Programming table `dp`.
    # dp[i] will store a tuple: (minimum total dissatisfaction for orders 1 to i, shipment day of the last batch which included order i).
    # Initialize all states with infinite dissatisfaction to signify they are initially unreachable / not yet computed.
    # Size is N+1 to accommodate indices 0 to N.
    dp = [(float('inf'), -1)] * (N + 1)
    
    # Base case: dp[0] represents the state before processing any orders.
    # The total dissatisfaction is 0. 
    # The 'last shipment day' is conceptually set to negative infinity.
    # This setup correctly handles the constraint S >= S_prev + X for the very first shipment.
    # When prev_S is -infinity, max(T_i, prev_S + X) correctly simplifies to T_i,
    # meaning the first shipment can occur as early as the placement day of the latest order in its batch.
    dp[0] = (0, -float('inf'))

    # Iterate through each order i from 1 to N to compute the optimal state dp[i].
    for i in range(1, N + 1):
        # Initialize the minimum dissatisfaction found so far for state i to infinity.
        min_D_i = float('inf') 
        # Initialize the shipment day corresponding to this minimum dissatisfaction.
        best_S_i = -1          
        
        # Consider all possible previous states dp[j] from which we could transition to state i.
        # A transition from dp[j] to dp[i] means the last shipment covers orders from j+1 to i.
        # The number of orders in this last shipment is (i - j).
        # Constraint: The number of orders per shipment must be between 1 and K (inclusive).
        # So, 1 <= i - j <= K. This implies i - K <= j <= i - 1.
        # Also, the index j must be non-negative (j >= 0) since it refers to a previous state dp[j].
        # Combining these conditions, the valid range for j is from max(0, i - K) up to i - 1.
        
        for j in range(max(0, i - K), i):
            # Retrieve the optimal results (minimum dissatisfaction `prev_D` and last shipment day `prev_S`) from state dp[j].
            prev_D, prev_S = dp[j] 
            
            # A basic check: If the state dp[j] was somehow unreachable (its dissatisfaction is infinite),
            # we cannot transition from it. Skip this j. This check is mostly for conceptual robustness.
            if prev_D == float('inf'):
                 continue

            # Determine the earliest possible shipment day (`current_S`) for the batch of orders j+1 to i.
            # Constraint 1: The shipment day must be on or after the placement day of the latest order in the batch, which is T_1based[i] (since T is sorted).
            # Constraint 2: The shipment day must be at least X days after the previous shipment day (`prev_S`).
            # The earliest possible day satisfying both is max(T_1based[i], prev_S + X).
            current_S = max(T_1based[i], prev_S + X)
            
            # Calculate the dissatisfaction generated specifically by shipping the current batch (orders j+1 to i) on day `current_S`.
            # Dissatisfaction for a single order k shipped on `current_S` is (current_S - T_k).
            # We need to sum this over all orders k in the current batch (from j+1 to i).
            # Sum_{k=j+1..i} (current_S - T_k) = (number of orders) * current_S - Sum_{k=j+1..i} T_k
            # Using prefix sums for efficiency: Sum_{k=j+1..i} T_k = P[i] - P[j].
            # Thus, the dissatisfaction for this shipment is (i - j) * current_S - (P[i] - P[j]).
            current_shipment_diss = (i - j) * current_S - (P[i] - P[j])
            
            # Calculate the total dissatisfaction to ship orders 1 to i if we choose to transition from state j.
            # Total dissatisfaction = (Dissatisfaction for first j orders) + (Dissatisfaction for current shipment).
            total_diss = prev_D + current_shipment_diss
            
            # Compare the calculated total dissatisfaction (`total_diss`) with the current minimum (`min_D_i`) found for state i.
            if total_diss < min_D_i:
                # If this path yields a lower total dissatisfaction, update `min_D_i` and the corresponding shipment day `best_S_i`.
                min_D_i = total_diss
                best_S_i = current_S
            elif total_diss == min_D_i:
                # Tie-breaking rule: If the total dissatisfaction is equal to the current minimum,
                # we choose the path that results in an earlier shipment day for the current batch (smaller `current_S`).
                # Rationale: An earlier shipment might allow subsequent shipments to also start earlier, potentially reducing future dissatisfaction.
                # So, update `best_S_i` to the minimum of its current value and `current_S`.
                best_S_i = min(best_S_i, current_S)

        # After checking all possible transitions (all valid j), store the computed optimal 
        # (minimum dissatisfaction `min_D_i`, corresponding shipment day `best_S_i`) for state i in the DP table.
        dp[i] = (min_D_i, best_S_i)

    # The final answer is the minimum total dissatisfaction required to ship all N orders.
    # This value is stored as the first element of the tuple in dp[N].
    print(dp[N][0])

# Execute the main function to solve the problem
solve()