import sys
from collections import deque

# Use a large negative number to represent unreachable states
# Maximum possible positive happiness is roughly N * (V * W) = 3000 * 10^9 * 3000 = 9e15.
# Max k*v-k^2 for one item type with weight w_i=1 is k=W, happ roughly W*V = 3000 * 10^9 = 3e12.
# Max total items sum k_i <= W.
# Maximum total happiness could be sum over items. If v_i is large, k_i could be small to get high happiness.
# Max total happiness is likely bounded. Perhaps sum (k_i v_i) where k_i w_i <= W.
# Worst case maybe k_i=1 for W types? No, sum k_i w_i <= W.
# If many items have w_i=1, we can take up to W items.
# Max happiness for one item type taking k items is approx (v/2)^2 if w_i is small enough.
# If v=10^9, (10^9/2)^2 = 2.5e17. Sum over N types? No, total weight constraint.
# The value range fits into signed 64-bit integer usually (~9e18). Python handles arbitrary int size.
# A value like -10**18 is sufficiently small compared to possible positive values.
INF = -10**18

def solve():
    N, W = map(int, sys.stdin.readline().split())
    items = []
    for _ in range(N):
        w, v = map(int, sys.stdin.readline().split())
        items.append((w, v))

    # dp[w] will store the maximum happiness for a total weight of exactly w
    # Initialize with INF for unreachable states, except dp[0] = 0
    dp = [INF] * (W + 1)
    dp[0] = 0

    for w_i, v_i in items:
        # Use a new DP table for the current item type to avoid using updated values
        # within the same iteration for different weights.
        next_dp = list(dp) 
        
        # Apply Convex Hull Trick optimization for this item type
        # Iterate through each remainder r modulo w_i
        for r in range(w_i):
            # Deque stores indices q_j such that points (q_j, dp[q_j * w_i + r] - q_j^2) form an upper convex hull
            # The points are (j, Y_j) where j is the quotient and Y_j = dp[j*w_i+r] - j^2.
            # We are looking for max_{0 <= j <= q} (Y_j + j * X) where X = 2*q - v_i.
            # Slopes are j (increasing), Query slopes X are 2*q - v_i (increasing with q).
            dq = deque()
            
            # Iterate through possible quotients q such that q * w_i + r <= W
            # j ranges from 0 up to q. The current q value represents the point being considered (j=q).
            for q in range((W - r) // w_i + 1):
                w = q * w_i + r # Current weight
                
                # --- Add point (q, dp[w] - q^2) to the deque ---
                # This point corresponds to using j=q items of the current type in the previous stage calculation.
                # We add point (q, dp[q*w_i + r] - q^2) based on the DP state *before* considering item i.
                # Only add if dp[w] from the previous state is reachable.
                if dp[w] > INF // 2: # Check if dp[w] is not the initial large negative value
                    point_j = q
                    point_y = dp[w] - point_j * point_j # Y coordinate for the point (j, Y_j) where j = point_j

                    # Maintain upper convex hull property (remove points that make the hull concave)
                    # Check points j1, j2, point_j from left to right (j1 < j2 < point_j)
                    # Remove j2 if slope(j1, j2) <= slope(j2, point_j)
                    # (y2 - y1) / (j2 - j1) <= (point_y - y2) / (point_j - j2)
                    # Cross product: (y2 - y1) * (point_j - j2) <= (point_y - y2) * (j2 - j1)
                    while len(dq) >= 2:
                        j1 = dq[-2]
                        j2 = dq[-1]
                        # Get y-coordinates for j1 and j2 from the *previous* dp state
                        y1 = dp[j1 * w_i + r] - j1 * j1
                        y2 = dp[j2 * w_i + r] - j2 * j2
                        
                        # Use cross product to check concavity
                        # (y_j2 - y_j1) * (point_j - j2) <= (point_y - y_j2) * (j2 - j1)
                        # Note: Correct comparison for upper convex hull with increasing x coordinates
                        # slope(j1, j2) <= slope(j2, point_j)
                        # (y2 - y1) * (point_j - j2) <= (point_y - y2) * (j2 - j1)
                        if (y2 - y1) * (point_j - j2) <= (point_y - y2) * (j2 - j1):
                             dq.pop()
                        else:
                             break
                    dq.append(point_j)

                # --- Query for optimal j <= q using slope X = 2*q - v_i ---
                # We want to find max_{j in dq} { (dp[j*w_i+r] - j^2) + j * X }
                # Since query slope X is increasing with q, we remove points from the front
                # whose slope to the next point is less than or equal to X.
                # Slope(j1, j2) <= X <=> (y2 - y1) / (j2 - j1) <= X
                # Cross product: (y2 - y1) <= X * (j2 - j1) (since j2 > j1)
                
                query_slope_X = 2 * q - v_i
                
                while len(dq) >= 2:
                    j1 = dq[0]
                    j2 = dq[1]
                    y1 = dp[j1 * w_i + r] - j1 * j1
                    y2 = dp[j2 * w_i + r] - j2 * j2
                    
                    # Check if slope(j1, j2) <= query_slope_X
                    # (y2 - y1) <= query_slope_X * (j2 - j1)
                    # This comparison needs care with negative X. The cross product comparison handles signs correctly.
                    # Check if the current best point j1 is worse than j2 for slope X.
                    # j1 is worse if y1 + j1*X <= y2 + j2*X
                    # y1 - y2 <= (j2 - j1) * X
                    if (y1 - y2) <= query_slope_X * (j2 - j1):
                         dq.popleft()
                    else:
                         break

                # If the deque is not empty, the optimal j for this q is the front element
                if len(dq) > 0:
                    j_star = dq[0]
                    # The number of items k of current type is q - j_star
                    k = q - j_star
                    
                    # Calculate the happiness achieved with j_star items of other types (weight j_star * w_i + r)
                    # and k items of the current type i
                    # This happiness is dp[j_star * w_i + r] + k * v_i - k^2
                    current_happiness = dp[j_star * w_i + r] + k * v_i - k * k
                    
                    # Update next_dp[w] with the maximum happiness for this weight
                    next_dp[w] = max(next_dp[w], current_happiness)
        
        # After processing all remainders for item i, update dp to next_dp
        dp = next_dp

    # The answer is the maximum value in the final dp table, considering weights up to W
    max_happiness = max(dp)

    # The maximum happiness must be at least 0 (by taking no items).
    # dp[0] = 0 initially ensures this if W >= 0.
    print(max(0, max_happiness))

solve()