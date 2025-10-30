import math
from typing import List

# Define helper functions gcd and lcm globally or within the class as static methods.
# Using global functions here.

def gcd(a, b):
    """Computes the greatest common divisor of a and b using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    # Return the absolute value to ensure GCD is non-negative.
    return abs(a) 

def lcm(a, b):
    """Computes the least common multiple of a and b."""
    # LCM is 0 if either number is 0.
    if a == 0 or b == 0:
        return 0
    # Calculate GCD first.
    common_divisor = gcd(a, b)
    # If GCD is 0, it implies both a and b are 0, which is handled above.
    # This check is for safety, though technically redundant if a, b != 0.
    if common_divisor == 0: return 0 
    # Calculate LCM using the formula: lcm(a, b) = abs(a * b) // gcd(a, b).
    # Python's arbitrary precision integers handle potentially large intermediate product abs(a * b).
    res = abs(a * b) // common_divisor
    return res

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        """
        Finds the k-th smallest amount that can be made using coins of given denominations.
        An amount can be formed if it is a multiple of any single coin denomination.
        We cannot combine coins of different denominations.

        Args:
            coins: A list of integers representing coin denominations.
            k: An integer representing the desired rank (k-th smallest).

        Returns:
            The k-th smallest amount that can be formed.
        """
        
        n = len(coins)
        
        # Pre-processing: Filter the coins list to remove any coin denomination `c`
        # if it is a multiple of another coin denomination `d` already present.
        # This is because any multiple of `c` is also a multiple of `d`, so `c` doesn't
        # contribute new unique amounts to the union set if `d` is already included.
        # Sorting helps in efficient filtering.
        coins.sort() 
        filtered_coins = []
        for coin_val in coins:
            is_multiple = False
            # Check if the current coin_val is a multiple of any coin already added to filtered_coins.
            for fc in filtered_coins:
                if coin_val % fc == 0:
                    is_multiple = True
                    break
            if not is_multiple:
                filtered_coins.append(coin_val)
        
        # Use the filtered list of coins for further calculation.
        coins = filtered_coins
        n = len(coins) # Update n to the size of the filtered list.
        
        # Helper function to calculate the count of distinct amounts less than or equal to X
        # that are multiples of at least one coin denomination in the `coins` list.
        # This uses the Principle of Inclusion-Exclusion.
        def calculate_count(X):
            count = 0
            # Iterate through all non-empty subsets of the `coins` list.
            # Each subset is represented by a bitmask `i` from 1 to 2^n - 1.
            for i in range(1, 1 << n):
                current_lcm = 1 # Initialize LCM for the current subset.
                
                # Compute the LCM of the coin denominations in the current subset.
                for j in range(n):
                    # Check if the j-th coin is included in the subset represented by mask `i`.
                    if (i >> j) & 1: 
                        # Update the LCM with the j-th coin.
                        current_lcm = lcm(current_lcm, coins[j])
                        
                        # Optimization: If the current LCM already exceeds X,
                        # any multiple of this LCM will also exceed X.
                        # Thus, this subset contributes 0 to the count for amounts <= X.
                        # Also, further LCM calculations with more coins will only result in a larger LCM.
                        # We can stop processing this subset early.
                        if current_lcm > X:
                           break # Exit the inner loop (over j).
                
                # After computing the LCM for the subset (or breaking early if LCM > X):
                # If the final computed LCM for the subset is within the bound X:
                if current_lcm <= X: 
                    # Determine the size of the current subset (number of coins included).
                    subset_size = bin(i).count('1')
                    
                    # Calculate how many multiples of `current_lcm` are less than or equal to X.
                    term = X // current_lcm
                    
                    # Apply the Inclusion-Exclusion Principle:
                    # Add the term if the subset size is odd.
                    # Subtract the term if the subset size is even.
                    if subset_size % 2 == 1: 
                        count += term
                    else: 
                        count -= term
            # Return the total count calculated using Inclusion-Exclusion.
            return count

        # Binary search for the k-th smallest amount.
        # The search space is [low, high].
        min_coin = coins[0] # The smallest coin denomination after filtering.
        low = 1 # The smallest possible amount could be 1 if 1 is a coin. Start search from 1.
        # A safe upper bound for the k-th smallest amount is k * min_coin.
        # The actual k-th amount cannot exceed this value.
        high = k * min_coin 
        ans = high # Initialize the answer candidate to the upper bound.

        # Perform binary search.
        while low <= high:
            mid = low + (high - low) // 2 # Calculate the middle value to check.
            
            # Mid=0 check: safety against edge cases, though low starts at 1.
            if mid == 0:
                 num_amounts = 0 # If mid is 0, count is 0.
            else:
                 # Calculate how many distinct amounts <= mid can be formed.
                 num_amounts = calculate_count(mid)
            
            # Check if the count `num_amounts` is sufficient.
            if num_amounts >= k:
                # If `mid` can generate at least `k` distinct amounts, then `mid` is a potential candidate
                # for the k-th smallest amount (or it might be larger than the k-th).
                # Store `mid` as the best answer found so far and try to find a smaller valid amount.
                # Search in the lower half [low, mid-1].
                ans = mid
                high = mid - 1 
            else: # num_amounts < k
                # If `mid` generates fewer than `k` amounts, it means `mid` is too small.
                # The k-th smallest amount must be larger than `mid`.
                # Search in the upper half [mid+1, high].
                low = mid + 1 
                
        # The loop terminates when low > high. `ans` holds the smallest value `mid`
        # for which `calculate_count(mid)` was >= k. This is the k-th smallest amount.
        return ans