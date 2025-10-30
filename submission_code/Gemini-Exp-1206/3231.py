class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        ans = 0
        reachable = 0
        i = 0
        while reachable < target:
            if i < len(coins) and coins[i] <= reachable + 1:
                reachable += coins[i]
                i += 1
            else:
                reachable += reachable + 1
                ans += 1
        return ans