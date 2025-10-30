from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        # The key observation is that to reach any position i,
        # you can always use the person with the smallest cost among persons 0...i.
        # That is because if you pay that person, then everyone behind
        # (i.e. with index greater than that person’s index) can swap for free.
        # Thus, the minimum cost needed to reach position i is just the minimum
        # value in cost[0...i]. We simply return the running minimum.

        n = len(cost)
        answer = [0] * n
        current_min = float('inf')
        for i in range(n):
            current_min = min(current_min, cost[i])
            answer[i] = current_min
        return answer