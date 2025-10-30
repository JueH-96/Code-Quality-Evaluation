import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        
        def calculate_lcm_upto_X(nums, X):
            """
            Calculates the Least Common Multiple (LCM) of a list of numbers.
            Returns X + 1 if the LCM exceeds X during calculation,
            to indicate that the true LCM is greater than X.
            This prevents overflow for very large LCMs and correctly
            signals that multiples of this LCM do not contribute to the count <= X.
            Constraints guarantee nums contains positive integers.
            """
            lcm_val = 1
            
            for num in nums:
                # Constraints guarantee num >= 1.
                g = math.gcd(lcm_val, num)
                
                # Calculate (lcm_val * num) // g without overflow.
                # This is equivalent to (lcm_val // g) * num.
                lcm_val_after_gcd = lcm_val // g

                # Check if the next LCM value `lcm_val_after_gcd * num` would exceed `X`.
                # We can check this using integer division: `lcm_val_after_gcd > X // num`.
                # This check works for positive integers `lcm_val_after_gcd`, `num`, and `X`.
                
                if lcm_val_after_gcd > X // num:
                    # The next LCM will be greater than X.
                    return X + 1 # Return a value > X
                    
                lcm_val = lcm_val_after_gcd * num

            return lcm_val

        def count_le(X, coins):
            """
            Counts the number of distinct positive integers <= X
            that are multiples of at least one coin in the given list.
            Uses the Principle of Inclusion-Exclusion.
            """
            n = len(coins)
            total_count = 0
            
            # Iterate through all non-empty subsets of coins using a bitmask.
            # A bitmask `i` from 1 to 2^n - 1 represents a subset.
            # If the j-th bit of `i` is set, coins[j] is in the subset.
            for i in range(1, 1 << n):
                subset_coins = []
                subset_size = 0
                
                # Build the subset of coins for the current mask `i`.
                for j in range(n):
                    if (i >> j) & 1:
                        subset_coins.append(coins[j])
                        subset_size += 1
                
                # Calculate the LCM of the coins in the current subset.
                # The function caps the LCM at X + 1 if it exceeds X.
                # X is guaranteed to be >= 1 in the binary search.
                current_lcm = calculate_lcm_upto_X(subset_coins, X)
                
                # The number of positive multiples of `current_lcm` that are <= X is X // current_lcm.
                # If `current_lcm` is X+1 (meaning the true LCM > X), then X // current_lcm is 0.
                term = X // current_lcm # Safe because current_lcm >= 1 (or X+1) and X >= 1.
                
                # Apply Inclusion-Exclusion principle:
                # Add the count if the subset size is odd, subtract if even.
                if subset_size % 2 == 1:
                    total_count += term
                else:
                    total_count -= term
                    
            return total_count

        # Binary search for the k-th smallest distinct amount.
        # We are searching for the minimum value `ans` such that `count_le(ans, coins) >= k`.

        # The smallest possible positive amount is 1.
        low = 1
        
        # A safe upper bound for the k-th smallest amount.
        # In the worst case (if only the largest coin existed), the k-th amount
        # would be k * max(coins).
        # k <= 2 * 10^9, max(coins) <= 25. Max value is approx 2e9 * 25 = 5e10.
        # A value slightly larger than this guarantees the answer is within the range.
        # 5 * 10**10 + 1 is a safe and reasonably tight upper bound.
        # Python's int handles this magnitude.
        high = 5 * 10**10 + 1 

        # Initialize the answer to a value that is definitely >= the true answer.
        # This allows the binary search logic (finding the minimum value satisfying the condition) to work.
        ans = high 

        # Perform binary search.
        while low <= high:
            # Calculate the middle value. Using (high - low) // 2 prevents potential overflow
            # if low and high were both very large, although Python 3 int handles this.
            mid = low + (high - low) // 2
            
            # Get the count of distinct amounts <= mid.
            count = count_le(mid, coins)
            
            # Check if the number of amounts <= mid is at least k.
            if count >= k:
                # If count >= k, it means `mid` is a potential candidate for the k-th
                # smallest amount, or the k-th smallest amount is actually smaller than `mid`.
                # We record `mid` as a potential answer and try searching in the lower half.
                ans = mid 
                high = mid - 1 
            else: # count < k
                # If count < k, it means `mid` is too small. The k-th smallest amount
                # must be larger than `mid`. We need to search in the upper half.
                low = mid + 1 
        
        # After the loop terminates, `ans` will hold the smallest value `X`
        # for which `count_le(X) >= k`, which is exactly the k-th smallest amount.
        return ans