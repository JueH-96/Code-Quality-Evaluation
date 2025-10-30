import math # Import math is not needed, it was included by mistake in thought process. Can be removed.
from typing import List

class Solution:
  """
  Solves the minimum added coins problem using a greedy approach.
  The strategy is to maintain the maximum reachable sum `reachable` such that all integers
  in `[1, reachable]` can be formed by summing subsequences of the coins considered so far (including added coins). 
  It iteratively extends this range using available coins from the input array `coins`
  or by adding new coins optimally until `reachable >= target`.
  """
  def minimumAddedCoins(self, coins: List[int], target: int) -> int:
    """
    Calculates the minimum number of coins to add to the array `coins` 
    such that every integer in the range [1, target] is obtainable 
    by summing a subsequence of the modified array.

    Args:
        coins: A list of integers representing the values of available coins.
                Constraints: 1 <= coins.length <= 10^5, 1 <= coins[i] <= target.
        target: The target integer up to which all sums should be obtainable.
                Constraints: 1 <= target <= 10^5.

    Returns:
        The minimum number of coins that need to be added.
    """
    
    # Sort the coins array to process coins in increasing order.
    # This is crucial for the greedy strategy because we want to use the smallest
    # available coins first to extend the reachable range.
    coins.sort()
    
    # `reachable` tracks the maximum sum such that all integers in the range `[1, reachable]` 
    # are obtainable. It starts at 0, signifying that initially, we cannot form any positive sum. 
    # The range [1, 0] is considered empty.
    reachable = 0
    
    # `added_coins` counts the number of coins we need to add to achieve the goal.
    added_coins = 0
    
    # `i` is the index pointer for iterating through the sorted `coins` array.
    i = 0
    
    # The main loop continues as long as our current `reachable` range does not cover `target`.
    # That is, while `reachable < target`.
    while reachable < target:
        # The next integer that we must be able to form is `reachable + 1`.
        
        # Check if the current coin `coins[i]` (if one exists) can help us reach `reachable + 1`.
        # This is possible if `coins[i]` is less than or equal to `reachable + 1`.
        if i < len(coins) and coins[i] <= reachable + 1:
            # If the current coin `coins[i]` is usable:
            # Using this coin extends the range of obtainable sums. If we could form sums
            # up to `reachable`, adding `coins[i]` allows us to form sums up to `reachable + coins[i]`.
            reachable += coins[i]
            # Move to the next coin in the sorted list.
            i += 1
        else:
            # If there are no more coins left (`i >= len(coins)`) or the current coin `coins[i]`
            # is too large (`coins[i] > reachable + 1`), we cannot form `reachable + 1` using the
            # available coins.
            
            # In this case, we must add a new coin. To cover `reachable + 1` and maximize
            # the extension of the reachable range with this single addition (greedy choice),
            # the optimal coin value to add is `reachable + 1`.
            
            # Adding a coin with value `reachable + 1` extends the maximum reachable sum to 
            # `reachable + (reachable + 1) = 2 * reachable + 1`.
            # Update `reachable` accordingly.
            reachable += (reachable + 1) 
            
            # Increment the count of added coins.
            added_coins += 1
            
            # IMPORTANT: We do not increment `i` here. The coin `coins[i]` (if it exists)
            # might become usable in the next iteration now that `reachable` has increased.
            # If `i` was already past the end of the `coins` list, the condition `i < len(coins)`
            # will continue to evaluate to false in subsequent iterations, correctly leading 
            # to adding more coins if `reachable` is still less than `target`.
                
    # Once the loop terminates (`reachable >= target`), it means we can form all integers
    # in the range `[1, target]`.
    # Return the total count of coins that were added.
    return added_coins