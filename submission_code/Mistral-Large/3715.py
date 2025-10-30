from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Step 1: Create an array to store the number of coins in each bag
        max_coordinate = max(r for l, r, _ in coins)
        bag_coins = [0] * (max_coordinate + 1)

        # Step 2: Populate the bag_coins array based on the given segments
        for l, r, c in coins:
            for i in range(l, r + 1):
                bag_coins[i] = c

        # Step 3: Use a sliding window to find the maximum sum of k consecutive bags
        current_sum = sum(bag_coins[:k])
        max_sum = current_sum

        for i in range(k, len(bag_coins)):
            current_sum += bag_coins[i] - bag_coins[i - k]
            max_sum = max(max_sum, current_sum)

        return max_sum

# Example usage:
# solution = Solution()
# print(solution.maximumCoins([[8,10,1],[1,3,2],[5,6,4]], 4))  # Output: 10
# print(solution.maximumCoins([[1,10,3]], 2))  # Output: 6