from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # We want to ensure that all values in [1, target] are obtainable.
        # Keep track of the current reachable range [1..reach].
        # Initially, reach = 0 (we can make no positive sum).
        # Sort the coins and iterate:
        #   If the next coin value c is <= reach+1, we can extend the reach by c.
        #   Otherwise, there's a gap at reach+1, so we "add" a coin of value reach+1,
        #   increment the count, and extend reach by reach+1.
        # Continue this process until we've considered all coins.
        # Finally, if reach < target, keep adding coins of value reach+1 until reach >= target.
        
        coins.sort()
        reach = 0
        added = 0
        i = 0
        n = len(coins)
        
        # While we haven't covered up to target:
        while reach < target:
            # If there's a coin we can use to extend the range without gap
            if i < n and coins[i] <= reach + 1:
                reach += coins[i]
                i += 1
            else:
                # We need to add a coin of value reach+1
                added += 1
                reach += (reach + 1)
        
        return added

# Example usage:
# sol = Solution()
# print(sol.minimumAddedCoins([1,4,10], 19))  # Output: 2
# print(sol.minimumAddedCoins([1,4,10,5,7,19], 19))  # Output: 1
# print(sol.minimumAddedCoins([1,1,1], 20))  # Output: 3