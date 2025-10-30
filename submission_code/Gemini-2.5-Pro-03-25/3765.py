import math
from collections import deque
from typing import List

class Solution:
    """
    Solves the minimum cost partitioning problem using Dynamic Programming
    optimized with Convex Hull Trick (CHT).
    
    Problem Description:
    Given arrays nums and cost, and integer k. Partition nums into subarrays.
    Cost of i-th subarray nums[l..r] is (Sum(nums[0..r]) + k * i) * Sum(cost[l..r]).
    Find the minimum total cost over all possible partitions.

    DP State Definition:
    Let dp[i][p] be the minimum cost to partition the prefix nums[0..i-1] into exactly p subarrays.
    The length of the prefix is i. The last element included is nums[i-1].

    Recurrence Relation:
    To compute dp[i][p], we consider all possible end points j-1 for the (p-1)-th subarray.
    The p-th subarray would then be nums[j..i-1].
    dp[i][p] = min_{0 <= j < i} { dp[j][p-1] + Cost(j, i-1, p) }
    
    Cost(j, i-1, p) is the cost of the p-th subarray nums[j..i-1]:
    Cost(j, i-1, p) = (Sum(nums[0..i-1]) + k * p) * Sum(cost[j..i-1])
                    = (P_N[i] + k * p) * (P_C[i] - P_C[j])
    Where P_N and P_C are 1-based prefix sums:
    P_N[x] = Sum(nums[0..x-1])
    P_C[x] = Sum(cost[0..x-1])
    
    The DP recurrence becomes:
    dp[i][p] = min_{0 <= j < i} { dp[j][p-1] + (P_N[i] + k*p) * (P_C[i] - P_C[j]) }
             = (P_N[i] + k*p) * P_C[i] + min_{0 <= j < i} { dp[j][p-1] - (P_N[i] + k*p) * P_C[j] }

    Convex Hull Trick Optimization:
    Let C = P_N[i] + k*p. This is the query slope for the CHT.
    We want to find the minimum of `Y_j - C * X_j` over points (X_j, Y_j) = (P_C[j], dp[j][p-1]).
    This standard form minimizes the intercept of lines with slope C passing through points (X_j, Y_j).
    Since the query slope C increases with i (for fixed p) because P_N[i] increases with i,
    and the points X_j = P_C[j] are added in increasing order of j (because P_C[j] increases with j),
    we can use a monotonic queue (deque) based CHT implementation for amortized O(1) updates and queries per step.

    Space Optimization:
    The computation of dp[i][p] only depends on values from dp[j][p-1].
    We can optimize space from O(N^2) to O(N) by using only two layers of the DP table:
    `prev_dp` stores values for p-1 subarrays, and `curr_dp` stores values for p subarrays.

    Time Complexity: O(N^2) - There are N iterations for p (number of subarrays) and N iterations for i (prefix length).
                     Each CHT operation (add point, query minimum) takes amortized O(1) time.
    Space Complexity: O(N) - For prefix sums P_N, P_C, DP arrays `prev_dp`, `curr_dp`, and the deque used for CHT.
    """
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        if n == 0: 
            return 0 # Handle edge case of empty input array

        # Calculate 1-based prefix sums for nums and cost arrays
        P_N = [0] * (n + 1)
        P_C = [0] * (n + 1)
        for i in range(n):
            P_N[i+1] = P_N[i] + nums[i]
            P_C[i+1] = P_C[i] + cost[i]

        # Initialize DP array for p=0 subarrays. dp[0][0] = 0 cost for empty prefix.
        # All other dp[i][0] are infinity since non-empty prefix needs at least one subarray.
        # `prev_dp` array stores the minimum costs for p-1 subarrays.
        prev_dp = [0] + [float('inf')] * n 
        
        # Variable to store the minimum total cost found for partitioning the entire array.
        min_total_cost = float('inf')

        # Check function for CHT convexity (lower envelope maintenance)
        # Returns True if point j2 is redundant (lies on or above the line segment j1-j3).
        # This maintains the lower envelope property where slopes of segments are strictly increasing.
        def check_convexity(j1, j2, j3):
            # Points are (P_C[j], prev_dp[j])
            Y1, Y2, Y3 = prev_dp[j1], prev_dp[j2], prev_dp[j3]
            X1, X2, X3 = P_C[j1], P_C[j2], P_C[j3]
            
            # Using cross product to avoid division and potential floating point inaccuracies.
            # Check if slope(j1, j2) >= slope(j2, j3)
            # Equivalent to checking if (Y2 - Y1) * (X3 - X2) >= (Y3 - Y2) * (X2 - X1)
            # Note: Python's arbitrary precision integers handle large intermediate values.
            return (Y2 - Y1) * (X3 - X2) >= (Y3 - Y2) * (X2 - X1)

        # Check function for CHT query slope comparison
        # Returns True if the slope of the segment (j1, j2) is less than or equal to query_slope C.
        # This is used to remove points from the front of the deque during query.
        def check_slope(j1, j2, C):
            Y1, Y2 = prev_dp[j1], prev_dp[j2]
            X1, X2 = P_C[j1], P_C[j2]

            # Check if slope(j1, j2) <= C using multiplication form
            # Equivalent to checking if (Y2 - Y1) <= C * (X2 - X1)
            return (Y2 - Y1) <= C * (X2 - X1)

        # Iterate through the number of subarrays p, from 1 to N
        for p in range(1, n + 1):
            # `curr_dp` array stores the minimum costs for p subarrays. Initialize with infinity.
            curr_dp = [float('inf')] * (n + 1)
            # Deque stores indices j representing points (P_C[j], prev_dp[j]) on the lower envelope.
            dq = deque()
            
            # Initialize the deque with the point corresponding to j=0, if it's reachable (finite cost).
            # For p=1, prev_dp[0]=dp[0][0]=0. For p>1, prev_dp[0]=dp[0][p-1]=inf.
            if prev_dp[0] != float('inf'):
                 dq.append(0)

            # Iterate through the end index i of the prefix nums[0..i-1]
            for i in range(1, n + 1):
                # Calculate the query slope C = P_N[i] + k * p for the CHT query.
                current_slope = P_N[i] + k * p
                
                # Query CHT: Remove points from the front of the deque which are suboptimal for the current_slope.
                # A point dq[0] is suboptimal if the segment dq[0]-dq[1] has slope <= current_slope.
                while len(dq) >= 2 and check_slope(dq[0], dq[1], current_slope):
                    dq.popleft()

                # Calculate the minimum cost dp[i][p] using the optimal j* found from CHT query.
                # The optimal j* is the index at the front of the deque (dq[0]).
                if dq:
                    j_star = dq[0]
                    # Calculate min_val = dp[j*][p-1] - C * P_C[j*] = prev_dp[j_star] - current_slope * P_C[j_star]
                    min_val = prev_dp[j_star] - current_slope * P_C[j_star]
                    # Calculate curr_dp[i] = dp[i][p] = C * P_C[i] + min_val
                    curr_dp[i] = current_slope * P_C[i] + min_val
                
                # Add point corresponding to index i to CHT structure for use in the *next* DP iteration (p+1)
                # The point added is (P_C[i], prev_dp[i]). Only add if prev_dp[i] is finite.
                if prev_dp[i] != float('inf'):
                    # Maintain convexity: Remove points from the back of the deque that violate the convexity property.
                    while len(dq) >= 2 and check_convexity(dq[-2], dq[-1], i):
                         dq.pop()
                    dq.append(i) # Add the new point index i

            # After computing costs for all prefixes ending at i for p subarrays,
            # update the overall minimum total cost if a partition of the whole array (N elements) is found.
            if curr_dp[n] != float('inf'):
                 min_total_cost = min(min_total_cost, curr_dp[n])
            
            # Update the DP state: the current DP results become the previous results for the next iteration (p+1).
            prev_dp = curr_dp
            
            # Early exit optimization: If all computed DP states for p subarrays are infinite,
            # it implies no valid partition into p subarrays exists. Consequently, no partition
            # with more than p subarrays can exist either. We can break the loop early.
            # Check if all prev_dp[1...N] are infinite. prev_dp[0] is handled separately.
            is_all_inf = True
            for idx in range(1, n + 1):
                 if prev_dp[idx] != float('inf'):
                      is_all_inf = False
                      break
            if is_all_inf: 
                break 

        # The problem constraints guarantee N >= 1 and positive costs/nums,
        # ensuring a valid partition exists and the minimum cost is finite.
        return min_total_cost