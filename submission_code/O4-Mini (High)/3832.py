from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        """
        The minimum cost to reach position i is simply the smallest cost[j] 
        among all j from 0 to i.  Reason: you can swap once with the cheapest
        person in front of or at i (pay cost[j]), landing at j, and then 
        move forward (to any k>j) for free until you reach i.
        """
        n = len(cost)
        ans = [0] * n
        minc = float('inf')
        for i in range(n):
            if cost[i] < minc:
                minc = cost[i]
            ans[i] = minc
        return ans