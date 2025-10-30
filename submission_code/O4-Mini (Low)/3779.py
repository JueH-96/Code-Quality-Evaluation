from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort the pizzas by weight
        pizzas.sort()
        n = len(pizzas)
        # Number of days = total pizzas / 4
        days = n // 4
        
        l, r = 0, n - 1
        total = 0
        
        for day in range(1, days + 1):
            if day & 1:  # odd day: gain the largest (Z)
                total += pizzas[r]
                # we've used up 1 largest and 3 smallest
                r -= 1
                l += 3
            else:        # even day: gain the second-largest (Y)
                # the two largest are at r-1 (Y) and r (Z)
                total += pizzas[r - 1]
                # we've used up those two largest and two smallest
                r -= 2
                l += 2
        
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxWeight([1,2,3,4,5,6,7,8]))  # Expected 14
    print(sol.maxWeight([2,1,1,1,1,1,1,1]))  # Expected 3