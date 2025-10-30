class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added = 0
        maxReachable = 0
        i = 0
        n = len(coins)
        
        # While we haven't covered up to target
        while maxReachable < target:
            # If there is a coin that can extend the reach, use it
            if i < n and coins[i] <= maxReachable + 1:
                maxReachable += coins[i]
                i += 1
            else:
                # Need to add a coin with value = maxReachable + 1
                added += 1
                maxReachable += (maxReachable + 1)
        return added