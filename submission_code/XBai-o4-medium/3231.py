class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_sum = 0
        index = 0
        res = 0
        while current_sum < target:
            if index < len(coins) and coins[index] <= current_sum + 1:
                current_sum += coins[index]
                index += 1
            else:
                # Add the coin (current_sum + 1)
                res += 1
                current_sum += current_sum + 1
        return res