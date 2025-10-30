from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        """
        Calculates the minimum number of coins needed to be added so that
        every integer from 1 to target is obtainable as a sum of a subsequence.

        Args:
            coins: A 0-indexed integer array representing the values of the coins.
            target: An integer representing the upper limit of the range [1, target].

        Returns:
            The minimum number of coins that need to be added.
        """
        # Sort the coins in ascending order. This is crucial for the greedy approach.
        coins.sort() 
        
        reachable = 0  # Represents the maximum sum currently obtainable starting from 1.
                       # We maintain the invariant that all sums in [0, reachable] are obtainable.
                       # (sum 0 is always obtainable by taking no coins).
                       # We aim to make sums [1, target] obtainable, which means we need
                       # reachable >= target.
        added_coins = 0 # Count of coins added
        coin_idx = 0 # Pointer to the current coin in the sorted coins array
        n = len(coins) # Number of original coins
        
        # We continue adding coins or using existing ones until
        # the maximum obtainable sum (reachable) is at least target.
        while reachable < target:
            # If we have coins left AND the current coin value is less than or equal to
            # the next sum we need to make (reachable + 1).
            # If we can reach all sums in [0, reachable] and we have a coin 'c'
            # where c <= reachable + 1, we can now form all sums from 0 to reachable + c.
            # The sums from c to reachable + c are formed by adding 'c' to sums from 0 to reachable.
            # Since c <= reachable + 1, the interval [c, reachable + c] overlaps or touches [1, reachable].
            # Their union with [0, reachable] is [0, reachable + c].
            # Thus, all sums in [1, reachable + c] are obtainable.
            if coin_idx < n and coins[coin_idx] <= reachable + 1:
                # Use the current coin. It extends the reachable range.
                reachable += coins[coin_idx]
                coin_idx += 1 # Move to the next coin
            # If we don't have coins left, OR the current coin value is too large
            # (coins[coin_idx] > reachable + 1), it means we cannot form
            # the sum 'reachable + 1' using the current set of coins
            # (or remaining coins). We have a gap starting at 'reachable + 1'.
            # We must add a coin to cover this gap and extend the range.
            # The most optimal coin to add is 'reachable + 1' itself.
            # Adding 'reachable + 1' to the sums we can already make [0, reachable]
            # allows us to form new sums in the range [(reachable + 1) + 0, (reachable + 1) + reachable]
            # = [reachable + 1, 2 * reachable + 1].
            # The union of [0, reachable] and [reachable + 1, 2 * reachable + 1] is [0, 2 * reachable + 1].
            # Thus, all sums in [1, 2 * reachable + 1] are obtainable.
            # This is the largest possible extension of the reachable range [1, reachable]
            # by adding a single coin when 'reachable + 1' is the minimum unreachable sum.
            else:
                # Add the coin 'reachable + 1'.
                reachable = 2 * reachable + 1
                added_coins += 1 # Increment the count of added coins
                # We do NOT increment coin_idx here, because the current coin
                # coins[coin_idx] was too large for the *previous* 'reachable + 1',
                # but it might be usable against the new, larger 'reachable + 1'
                # in the next iteration. We only use coins[coin_idx] when it is
                # small enough (<= current reachable + 1).
        
        return added_coins