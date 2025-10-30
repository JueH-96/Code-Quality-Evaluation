from typing import List
from collections import deque

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)

        # Calculate prefix sums
        # P_nums[i] = sum(nums[0...i-1]) for i > 0, P_nums[0] = 0
        # P_cost[i] = sum(cost[0...i-1]) for i > 0, P_cost[0] = 0
        P_nums = [0] * (n + 1)
        P_cost = [0] * (n + 1)
        for i in range(n):
            P_nums[i + 1] = P_nums[i] + nums[i]
            P_cost[i + 1] = P_cost[i] + cost[i]

        # dp[i] = minimum cost to divide nums[0...i-1] into p-1 segments (from previous iteration of p)
        # new_dp[i] = minimum cost to divide nums[0...i-1] into p segments (current iteration of p)

        # Initialize dp for p=0 (0 segments)
        # Cost is 0 for an empty prefix with 0 segments
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        # Overall minimum cost found so far
        min_total_cost = float('inf')

        # Iterate over the number of segments p (from 1 to n)
        # We need to find the minimum cost for dividing the whole array nums[0...n-1]
        # into p segments, for each p from 1 to n.
        # The result is the minimum of these costs.
        for p in range(1, n + 1):
            new_dp = [float('inf')] * (n + 1)
            dq = deque() # Deque stores indices j representing split points

            # Iterate over the end index i of the current p-th segment (0-indexed end index is i-1)
            # dp_table[p][i] = min cost for nums[0...i-1] into p segments.
            # To form p segments ending at i-1, the last segment is nums[j...i-1]
            # where nums[0...j-1] was divided into p-1 segments.
            # The split point index j ranges from p-1 up to i-1.
            # The length of the prefix nums[0...i-1] is i.
            # This prefix must contain at least p segments, so i >= p.

            for i in range(p, n + 1):
                # At this step i, we are calculating new_dp[i] (cost for prefix i-1 using p segments).
                # The split point j goes from p-1 to i-1.
                # The point corresponding to split index i-1 is (P_cost[i-1], dp[i-1]).
                # This point becomes available for queries from index i onwards.
                # We need to add this point (split index j = i-1) to the CHT before querying for i.
                j_add = i - 1
                
                # Check if dp[j_add] from the previous layer (p-1 segments) is finite.
                # j_add = i-1 must be achievable with p-1 segments. Min length for p-1 segments is p-1.
                # So, j_add >= p-1 must hold. This is guaranteed by i >= p => i-1 >= p-1.
                if dp[j_add] != float('inf'):
                     x3, y3 = P_cost[j_add], dp[j_add]
                     
                     # Maintain lower convex hull property in deque by removing redundant points from the back.
                     # Remove j2 (dq[-1]) if slope(j1, j2) >= slope(j1, j_add), where j1 = dq[-2].
                     # (y2 - y1) / (x2 - x1) >= (y3 - y1) / (x3 - x1) ?
                     # Using cross product to avoid float and maintain precision:
                     # (y2 - y1) * (x3 - x1) >= (y3 - y1) * (x2 - x1)
                     # Python's int handles arbitrary size, so no explicit 128-bit needed.
                     while len(dq) >= 2:
                          j1 = dq[-2]
                          j2 = dq[-1]
                          x1, y1 = P_cost[j1], dp[j1]
                          x2, y2 = P_cost[j2], dp[j2]
                          
                          # Check redundancy: is point j2 above or on the line segment (j1, j_add)?
                          # Handle potential vertical lines if P_cost is not strictly increasing.
                          # P_cost is strictly increasing for index >= 1 (since cost[i] >= 1). P_cost[0]=0.
                          # Indices in dq are j >= p-1.
                          # If p=1, indices j are 0, 1, ... P_cost is strictly increasing for j > 0.
                          # If p>=2, indices j are >= 1. P_cost is strictly increasing.
                          # So x1 <= x2 <= x3 with strict inequality if indices are distinct and > 0.
                          # If x1 == x2, it implies j1 == j2 == 0, which can only happen if len(dq) < 2.
                          # If x2 == x3, it implies j2 == j_add. This is handled by checking dq[-1].
                          
                          # Standard cross product check assuming x1 < x2 and x1 < x3
                          # (y2 - y1) * (x3 - x1) >= (y3 - y1) * (x2 - x1)
                          # This checks if slope(j1, j2) >= slope(j1, j_add)
                          # Use this form as it handles potential non-strictly increasing P_cost correctly with signed values.
                          # However, if x1==x2 or x2==x3, this form needs care or other checks.
                          # Given P_cost property, x1 < x2 and x2 < x3 holds if j1 < j2 < j_add and j1 >= 1.
                          # The check `(y2 - y1) * (x3 - x1) >= (y3 - y1) * (x2 - x1)` is the correct condition for removing j2 when adding j_add to the end for lower hull CHT with increasing x coordinates.
                          # It checks if the slope of line j1->j2 is greater than or equal to the slope of line j1->j_add.
                          if (y2 - y1) * (x3 - x1) >= (y3 - y1) * (x2 - x1):
                               dq.pop()
                          else:
                               break
                     
                     # Add j_add to the deque if it's not redundant due to having the same x-coordinate
                     # and a worse or equal y-coordinate compared to the last point in dq.
                     # Since P_cost[j] is non-decreasing with j, P_cost[j_add] >= P_cost[dq[-1]] holds.
                     # Add if x is strictly greater OR (x is same AND y is strictly less).
                     if len(dq) == 0 or P_cost[j_add] > P_cost[dq[-1]] or (P_cost[j_add] == P_cost[dq[-1]] and dp[j_add] < dp[dq[-1]]):
                          dq.append(j_add)
                     # else: if P_cost[j_add] == P_cost[dq[-1]] and dp[j_add] >= dp[dq[-1]], j_add is redundant

                # Query CHT with slope m = P_nums[i] + k * p
                # We want to minimize dp[j] - m * P_cost[j] over j in deque where p-1 <= j < i.
                # The indices in dq are j >= p-1 and are added up to i-1, so all j in dq satisfy j <= i-1.
                # The slopes m = P_nums[i] + k * p are increasing with i.
                # This is a standard CHT query for increasing slopes.
                # Remove points from front of deque while slope(dq[0], dq[1]) <= m.
                # Need slope(j1, j2) <= m
                # (y2 - y1) / (x2 - x1) <= m
                # (y2 - y1) <= m * (x2 - x1) # since x2 - x1 > 0 for j1 < j2 (unless j1=j2=0 which implies len<2)

                while len(dq) >= 2:
                    j1 = dq[0]
                    j2 = dq[1]
                    x1, y1 = P_cost[j1], dp[j1]
                    x2, y2 = P_cost[j2], dp[j2]
                    
                    # Check if slope(j1, j2) <= m
                    # (y2 - y1) <= m * (x2 - x1)
                    # Use Python int for large number multiplication
                    if (y2 - y1) <= m * (x2 - x1):
                        dq.popleft()
                    else:
                        break
                
                # The optimal split point j is dq[0].
                # Check if deque is not empty before accessing dq[0].
                # dq can be empty if no valid split point j (from p-1 to i-1) had a finite dp[j].
                if dq:
                    best_j = dq[0]
                    
                    # Calculate the cost for dp[p][i]
                    # Cost = (sum(nums[0..i-1]) + k * p) * sum(cost[best_j..i-1]) + dp[best_j]
                    # Cost = (P_nums[i] + k * p) * (P_cost[i] - P_cost[best_j]) + dp[best_j]
                    # This is equivalent to (P_nums[i] + k * p) * P_cost[i] + (dp[best_j] - (P_nums[i] + k * p) * P_cost[best_j])
                    
                    m = P_nums[i] + k * p # Recalculate m just to be safe, although it's the same as the query slope
                    min_val = dp[best_j] - m * P_cost[best_j]
                    new_dp[i] = (P_nums[i] + k * p) * P_cost[i] + min_val

            # After computing new_dp for all i in [p, n], the dp array for the next iteration p+1 is new_dp.
            dp = new_dp 
            
            # After computing the minimum cost to divide nums[0...n-1] into exactly p segments (new_dp[n]),
            # update the overall minimum total cost found so far.
            min_total_cost = min(min_total_cost, dp[n])

        return min_total_cost