from typing import List

class Solution:
  def minimumAddedCoins(self, coins: List[int], target: int) -> int:
    """
    Calculates the minimum number of coins to add to an array so that
    every integer in the range [1, target] is obtainable.
    """
    coins.sort()
    
    reachable_sum = 0  # Represents the maximum sum k such that all integers in [1, k] are obtainable.
                       # Initially 0, meaning no positive integer is obtainable yet.
    coins_added = 0
    i = 0  # Pointer for the sorted coins array
    
    while reachable_sum < target:
      # Check if the current smallest unused coin from the input array can be used
      if i < len(coins) and coins[i] <= reachable_sum + 1:
        # If coins[i] is small enough (i.e., <= reachable_sum + 1),
        # we can use it to extend our range of obtainable sums.
        # All sums up to reachable_sum + coins[i] are now obtainable.
        reachable_sum += coins[i]
        i += 1  # Move to the next coin in the input array
      else:
        # If there are no more coins in the input array (i >= len(coins)),
        # or if the current coin coins[i] is too large (coins[i] > reachable_sum + 1),
        # we cannot form reachable_sum + 1 with the current set of coins.
        # We must add a new coin.
        # To cover reachable_sum + 1 and extend our reach as much as possible,
        # we add a coin of value (reachable_sum + 1).
        new_coin_value = reachable_sum + 1
        reachable_sum += new_coin_value
        coins_added += 1
        # We do not increment i here, as coins[i] (if it exists) has not been used yet
        # and should be considered in the next iteration with the updated reachable_sum.
            
    return coins_added