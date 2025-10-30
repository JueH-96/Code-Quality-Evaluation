from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # current[i] will track the minimum cost to obtain type (i) from index i
        # after considering up to t shifts
        current = nums[:]  
        
        # Best cost when we do not shift at all (t = 0)
        best = sum(current)
        
        # Try all possible total number of shifts t from 1 to n-1
        for t in range(1, n):
            # Update the cost array based on the formula:
            # cost[t][i] = min(cost[t-1][i], nums[(i - t) mod n])
            # which reflects possibly picking the chocolate originally at (i - t)
            # after t shifts, if that is cheaper than the already-known cost.
            for i in range(n):
                current[i] = min(current[i], nums[(i - t) % n])
            
            # The total cost is the sum of the updated "current" array plus the cost for t shifts
            cost = sum(current) + t * x
            best = min(best, cost)
        
        return best