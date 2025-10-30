from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort()
        n = len(pizzas)
        groups = n // 4
        # Determine how many days have odd and even indices.
        odd_count = (groups + 1) // 2  # days: 1,3,5,...
        even_count = groups // 2       # days: 2,4,6,...
        gain = 0
        low = 0
        high = n - 1
        
        # For odd days, you gain the largest pizza in the group.
        # We form groups by taking one largest pizza and three smallest pizzas.
        for _ in range(odd_count):
            # choose largest available as Z
            gain += pizzas[high]
            high -= 1
            # Remove the three smallest pizzas (they serve as the lower three: W, X, Y)
            low += 3
        
        # For even days, you gain the second-largest in your chosen group.
        # To maximize that, for even day groups we take two largest ones and two smallest ones.
        for _ in range(even_count):
            # The two largest available:
            Y = pizzas[high - 1]  # second largest in the group
            gain += Y
            # Remove these two largest pizzas
            high -= 2
            # Remove two smallest pizzas for the rest of the group
            low += 2

        return gain

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    pizzas1 = [1,2,3,4,5,6,7,8]
    print(sol.maxWeight(pizzas1))  # Expected output: 14
    
    # Example 2
    pizzas2 = [2,1,1,1,1,1,1,1]
    print(sol.maxWeight(pizzas2))  # Expected output: 3