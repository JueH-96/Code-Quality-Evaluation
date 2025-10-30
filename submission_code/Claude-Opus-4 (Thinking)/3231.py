class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reachable = 0
        count = 0
        i = 0
        
        while max_reachable < target:
            if i < len(coins) and coins[i] <= max_reachable + 1:
                # Use existing coin to extend range
                max_reachable += coins[i]
                i += 1
            else:
                # Need to add a coin with value max_reachable + 1
                max_reachable += max_reachable + 1
                count += 1
        
        return count