from typing import List
import sys

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        
        # dp[j] will store the minimum cost to achieve a total capacity of exactly j.
        # Capacity is defined as the sum of (1 + time[i]) for walls painted by the paid painter.
        # We need the total capacity to be at least n.
        # The minimum total capacity required is n.
        # When we choose a wall i with the paid painter, it costs cost[i] and
        # contributes a capacity of (1 + time[i]) to the total.
        # The free painter can paint walls as long as the paid painter is busy.
        # If the paid painter spends a total time T_paid painting N_paid walls,
        # the free painter can paint up to T_paid walls.
        # The total number of walls painted is N_paid + N_free. We need N_paid + N_free = n.
        # The constraint is N_free <= T_paid.
        # Substituting N_free = n - N_paid, we get n - N_paid <= T_paid,
        # which means n <= N_paid + T_paid.
        # N_paid is the number of walls painted by the paid painter, sum(1 for paid walls).
        # T_paid is the total time spent by the paid painter, sum(time[i] for paid walls).
        # So the condition becomes n <= sum(1 for paid walls) + sum(time[i] for paid walls)
        # n <= sum(1 + time[i] for paid walls).
        # This means the sum of (1 + time[i]) for the walls painted by the paid painter must be at least n.
        # This is a 0/1 knapsack type problem where items are walls, weight is cost[i],
        # and value (or capacity contribution) is 1 + time[i]. We want to achieve a minimum
        # total value of n with minimum total weight (cost).

        # Let dp[j] be the minimum cost to achieve a total capacity of exactly j.
        # The base case is dp[0] = 0 (achieve capacity 0 with cost 0 by painting no walls paid).
        # The maximum relevant capacity we need to track is n + max(time[i]).
        # If we are at capacity k < n and add a wall with value v = 1 + time[i],
        # the new capacity is k + v. The maximum this can be is (n-1) + (1 + max(time[i])) = n + max(time[i]).
        # Any capacity greater than n + max(time[i]) achieved from a state < n would require
        # a single wall with time > max(time[i]), which is impossible.
        # Any capacity greater than n + max(time[i]) achieved from a state >= n would be
        # suboptimal in cost (since costs are positive), so we don't need to track them specifically.
        # Max time[i] is 500, so max value is 501. Max relevant capacity is n + 500.
        # We need a DP array of size up to index n + 500. Size is n + 500 + 1.
        
        max_relevant_capacity = n + 500
        dp_size = max_relevant_capacity + 1
        
        # Initialize DP table with infinity. Use a large float value.
        dp = [float('inf')] * dp_size
        
        # Base case: Capacity 0 achieved with cost 0.
        dp[0] = 0
        
        # Iterate through each wall (item)
        for i in range(n):
            c = cost[i]
            t = time[i]
            val = t + 1 # Capacity contributed by this wall if painted by paid painter
            
            # Update DP table: Consider using the paid painter for wall i.
            # This is a 0/1 knapsack update. To compute dp[j] (min cost for exactly capacity j)
            # after considering wall i, we can either not use wall i (cost dp_before[j])
            # or use wall i (cost dp_before[j - val] + c).
            # dp_after[j] = min(dp_before[j], dp_before[j - val] + c).
            # By iterating the capacity j downwards, we ensure that when we calculate dp[j],
            # the value dp[j - val] still holds the minimum cost achieved *before* considering wall i.
            
            # Iterate downwards from the maximum possible capacity index.
            # We only need to consider capacities that are reachable.
            # The current capacity j can be reached by adding wall i (value val)
            # to a previous capacity j - val. We need j - val >= 0.
            # The loop range j from max_relevant_capacity down to val ensures j - val >= 0
            # is always true within the valid index range [val, max_relevant_capacity].
            # Iterating down to 0 and checking `j - val >= 0` is equivalent and arguably cleaner.
            
            for j in range(max_relevant_capacity, -1, -1):
                 # If the previous capacity state `j - val` is valid (index >= 0)
                 # and was reachable (cost not infinity),
                 # we can transition from capacity `j - val` to capacity `j` by adding wall `i`.
                 if j - val >= 0 and dp[j - val] != float('inf'):
                     # The new potential cost to reach capacity `j` is `dp[j - val] + c`.
                     # Update the minimum cost for capacity `j`.
                     dp[j] = min(dp[j], dp[j - val] + c)
                     
        # After processing all walls, the minimum cost to paint n walls is the minimum cost
        # to achieve a total capacity of at least n.
        # This corresponds to the minimum value in dp[k] for all k >= n.
        # Since we tracked capacities up to max_relevant_capacity, the answer is
        # the minimum value in dp[k] for k from n up to max_relevant_capacity.
        
        min_cost = float('inf')
        
        # Iterate through all capacity states that are >= n to find the minimum cost.
        # These states range from index n up to the maximum index in the DP array.
        for j in range(n, dp_size):
            min_cost = min(min_cost, dp[j])
            
        return min_cost