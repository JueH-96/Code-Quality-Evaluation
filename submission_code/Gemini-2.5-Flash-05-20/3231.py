from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # Sort the coins to process them in increasing order.
        # This is essential for the greedy strategy to work correctly,
        # as we want to use the smallest possible existing coins to extend
        # our reachable sum range.
        coins.sort()

        added_coins_count = 0
        
        # current_max_sum represents the maximum sum X such that all integers
        # from 1 to X are currently obtainable using the coins considered so far.
        # Initially, only sum 0 (by picking no coins) is obtainable.
        current_max_sum = 0
        
        # Pointer for iterating through the sorted coins array.
        i = 0
        n = len(coins)

        # The loop continues as long as we haven't reached our target sum.
        # We need to ensure all sums up to 'target' are obtainable.
        while current_max_sum < target:
            # Check if we have an existing coin that can help extend the range.
            # An existing coin 'coins[i]' is useful if it's small enough:
            # it must be less than or equal to `current_max_sum + 1`.
            # If `current_max_sum + 1` is the smallest sum we cannot form,
            # and `coins[i]` is small enough, we can use it to cover `current_max_sum + 1`
            # and extend our range to `current_max_sum + coins[i]`.
            if i < n and coins[i] <= current_max_sum + 1:
                current_max_sum += coins[i]
                i += 1
            else:
                # If we don't have an existing coin that is small enough (either
                # `i >= n` meaning no more coins, or `coins[i] > current_max_sum + 1`),
                # it means there's a gap. The smallest sum we cannot form is
                # `current_max_sum + 1`.
                # To cover this gap and maximize the extension of our sum range,
                # we must add the coin `current_max_sum + 1`.
                # By adding `current_max_sum + 1`, we can now form all sums from
                # 0 up to `current_max_sum + (current_max_sum + 1)`.
                added_coins_count += 1
                current_max_sum += (current_max_sum + 1)
        
        # Once current_max_sum is >= target, we have successfully covered all
        # sums from 1 to target.
        return added_coins_count