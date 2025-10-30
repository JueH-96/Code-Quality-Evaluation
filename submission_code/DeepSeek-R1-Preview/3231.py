class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current = 0
        res = 0
        index = 0
        n = len(coins)
        while current < target:
            if index < n and coins[index] <= current + 1:
                current += coins[index]
                index += 1
            else:
                res += 1
                current += current + 1
        return res