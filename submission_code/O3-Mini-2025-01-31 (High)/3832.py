from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        # The key observation is that once you pay to swap with a person,
        # you can then swap with any person ahead for free because they become “behind” you.
        # Therefore, to reach any position i (which is one of the people ahead of the end),
        # it suffices to pay the minimum cost among persons 0 to i.
        #
        # For example, consider cost = [5, 3, 4, 1, 3, 2]:
        # For i = 0, the best (and only) choice is to swap with person 0, costing 5.
        # For i = 1, the best is to swap with person 1, costing 3 (cheaper than 5).
        # For i = 2, even though person 2 costs 4, you can instead use person 1's swap (cost 3)
        # and then swap with person 2 for free. So the cost remains min(5, 3, 4) which is 3.
        #
        # Thus, answer[i] is simply the minimum value in cost[:i+1] for each i.
        
        ans = []
        min_cost = float('inf')
        for c in cost:
            min_cost = min(min_cost, c)
            ans.append(min_cost)
        return ans