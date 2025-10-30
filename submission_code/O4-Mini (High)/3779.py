from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort the pizzas by weight
        pizzas.sort()
        n = len(pizzas)
        
        # Two pointers to simulate removing from both ends
        l, r = 0, n - 1
        
        ans = 0
        days = n // 4
        
        # Day index i: 0-based, so i=0 is day 1 (odd), i=1 is day 2 (even), etc.
        for i in range(days):
            if i % 2 == 0:
                # Odd-numbered day: gain the heaviest remaining pizza
                ans += pizzas[r]
                # Remove that pizza and use up 3 lightest as fillers
                r -= 1
                l += 3
            else:
                # Even-numbered day: gain the second-heaviest remaining pizza
                ans += pizzas[r - 1]
                # Remove the heaviest (unused) and the second-heaviest (gained),
                # then use up 2 lightest as fillers
                r -= 2
                l += 2
        
        return ans