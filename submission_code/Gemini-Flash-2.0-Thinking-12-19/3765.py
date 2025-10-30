import sys
from typing import List
from collections import deque

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)

        # Calculate prefix sums
        S_n = [0] * (n + 1)
        S_c = [0] * (n + 1)
        for i in range(n):
            S_n[i + 1] = S_n[i] + nums[i]
            S_c[i + 1] = S_c[i] + cost[i]

        # dp[j][i]: minimum cost to partition nums[0...i-1] into exactly j subarrays
        # dp_full[j][i] where j is the number of subarrays, i is the length of the prefix nums[0...i-1]
        # 0 <= j <= i <= n
        
        # Initialize DP table with infinity
        dp_full = [[float('inf')] * (n + 1) for _ in range(n + 1)]

        # Base case: 0 subarrays partition 0 elements with cost 0
        dp_full[0][0] = 0

        # Iterate through the number of subarrays j (from 1 to n)
        for j in range(1, n + 1):
            # CHT deque for layer j. Stores indices p from the previous layer (j-1).
            # Points are (S_c[p], dp[j-1][p]).
            # Query slope is m = -(S_n[i] + k*j) which depends on the current prefix length i.
            # Slopes m are non-increasing as i increases.
            # x-coordinates S_c[p] are strictly increasing as p increases (since cost[p] >= 1).
            # This fits standard CHT with increasing x and non-increasing query slopes.
            
            dq = deque() # Stores indices p

            # Iterate through the end index (exclusive) of the prefix i (from j to n)
            # Compute dp_full[j][i], representing partition of nums[0...i-1]
            for i in range(j, n + 1):
                # The split point p ranges from j-1 to i-1.
                # The previous state is dp[j-1][p], corresponding to partitioning nums[0...p-1].
                # The point (S_c[p], dp[j-1][p]) becomes available for CHT queries when p is known.
                # When we are calculating dp[j][i], the point with index p = i-1 is the latest available point from layer j-1.
                # So, we add point p = i-1 (if valid) before calculating dp_full[j][i].
                
                p_to_add = i - 1
                
                # Check if p_to_add is a valid index for dp[j-1] and if dp[j-1][p_to_add] is finite.
                # p_to_add must be >= j-1 for dp[j-1][p_to_add] to be meaningful (at least j-1 elements for j-1 subarrays).
                if p_to_add >= j - 1 and dp_full[j-1][p_to_add] < float('inf'):
                    x_p_add = S_c[p_to_add]
                    y_p_add = dp_full[j-1][p_to_add]

                    # Maintain lower convex hull by removing points from the back
                    # Check if slope(p1, p2) >= slope(p2, p_to_add)
                    # Use cross product to avoid division: (y2 - y1) * (x_p_add - x2) >= (y_p_add - y2) * (x2 - x1)
                    while len(dq) >= 2:
                        p1 = dq[-2]
                        p2 = dq[-1]
                        x1 = S_c[p1]
                        y1 = dp_full[j-1][p1]
                        x2 = S_c[p2]
                        y2 = dp_full[j-1][p2]

                        # Using standard Python integers which handle large numbers
                        if (y2 - y1) * (x_p_add - x2) >= (y_p_add - y2) * (x2 - x1):
                            dq.pop()
                        else:
                            break
                    dq.append(p_to_add)

                # Calculate dp_full[j][i]. Query with slope m = -(S_n[i] + k * j).
                # The relevant split points p are in the range [j-1, i-1].
                # The points currently in the deque are exactly (a subset of) the optimal points from this range.
                
                if len(dq) > 0:
                    m = -(S_n[i] + k * j)

                    # Query the deque for the optimal point p that minimizes (dp[j-1][p] - m * S_c[p])
                    # or dp[j-1][p] + m * S_c[p]
                    # For decreasing slopes m, the optimal point moves towards the right in the deque.
                    # Remove points from the front that are no longer optimal.
                    # A point p1 is suboptimal compared to p2 (next in deque) if y1 + m*x1 >= y2 + m*x2
                    # y1 - y2 >= m * (x2 - x1)
                    # (y1 - y2) / (x2 - x1) >= m  (since x2 > x1)
                    # So, remove p1 if slope'(p1, p2) >= m where slope'(p1, p2) = (y1 - y2) / (x2 - x1).
                    # Use cross product: (y1 - y2) >= m * (x2 - x1)
                    
                    while len(dq) >= 2:
                        p1 = dq[0]
                        p2 = dq[1]
                        x1 = S_c[p1]
                        y1 = dp_full[j-1][p1]
                        x2 = S_c[p2]
                        y2 = dp_full[j-1][p2]

                        if (y1 - y2) >= m * (x2 - x1):
                             dq.popleft()
                        else:
                            break

                    # The optimal split point p is at the front of the deque
                    optimal_p = dq[0]

                    # Calculate dp_full[j][i] using the optimal split point optimal_p
                    # dp[j][i] = dp[j-1][optimal_p] + (S_n[i] + k * j) * (S_c[i] - S_c[optimal_p])
                    dp_full[j][i] = dp_full[j-1][optimal_p] + (S_n[i] + k * j) * (S_c[i] - S_c[optimal_p])

        # The minimum total cost is the minimum cost to partition the whole array nums[0...n-1]
        # using any number of subarrays from 1 to n.
        # This corresponds to min(dp_full[j][n]) for j from 1 to n.
        min_total_cost = float('inf')
        for j in range(1, n + 1):
            min_total_cost = min(min_total_cost, dp_full[j][n])

        return min_total_cost