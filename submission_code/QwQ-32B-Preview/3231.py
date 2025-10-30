class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reach = 0
        added = 0
        index = 0
        n = len(coins)
        while max_reach < target:
            if index < n and coins[index] <= max_reach + 1:
                max_reach += coins[index]
                index += 1
            else:
                # Need to add a coin with value max_reach + 1
                added += 1
                max_reach += max_reach + 1
        return added