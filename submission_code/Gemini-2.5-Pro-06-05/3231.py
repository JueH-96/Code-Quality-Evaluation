from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        """
        Calculates the minimum number of coins to add to the array so that every
        integer in the range [1, target] is obtainable.
        """
        
        # Sort the coins to process them in increasing order. This is key to the
        # greedy approach.
        coins.sort()
        
        # 'reachable' is the maximum integer 'k' such that all integers
        # in the range [1, k] can be formed by a sum of a subsequence of the
        # coins considered so far (including the ones we've added).
        # Initially, we can't form any positive integer, so reachable is 0.
        reachable = 0
        
        # 'added_coins' counts the number of coins we need to add.
        added_coins = 0
        
        # 'i' is an index to iterate through the sorted 'coins' array.
        i = 0
        
        # We continue building our set of obtainable sums until we can form
        # every integer up to 'target'.
        while reachable < target:
            # The next integer we need to be able to form is `reachable + 1`.
            
            # Case 1: Use an existing coin.
            # If we have an available coin and its value is less than or equal to
            # `reachable + 1`, we can use it to extend our range without a gap.
            if i < len(coins) and coins[i] <= reachable + 1:
                # By using this coin, the new reachable range is [1, reachable + coins[i]].
                reachable += coins[i]
                i += 1
            else:
                # Case 2: Add a new coin.
                # This occurs if the next coin is too large or we've used all coins.
                # We must add a coin to form `reachable + 1`. The optimal choice is
                # to add a coin of value `reachable + 1`.
                
                added_coins += 1
                
                # Update `reachable` by adding the new coin's value.
                # The new range becomes [1, reachable + (reachable + 1)].
                reachable += (reachable + 1)
                
        return added_coins