class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        max_reach = 0
        res = 0
        i = 0
        n = len(coins)
        
        while max_reach < target:
            # Process all coins that can extend the current max_reach
            while i < n and coins[i] <= max_reach + 1:
                max_reach += coins[i]
                i += 1
            if max_reach >= target:
                break
            # Add the coin (max_reach + 1) to fill the gap
            res += 1
            added_coin = max_reach + 1
            max_reach += added_coin
        
        return res