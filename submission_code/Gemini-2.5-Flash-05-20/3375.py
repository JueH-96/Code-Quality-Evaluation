from typing import List

class Solution:
    # Helper function for Greatest Common Divisor
    def _gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    # Helper function for Least Common Multiple, capped at a limit X.
    # If the calculated LCM exceeds X, it returns X + 1 to signify "too large".
    def _safe_lcm(self, a: int, b: int, X: int) -> int:
        # coin values are positive, so a and b will be positive
        
        # If one of the numbers is already greater than X, their LCM will also be greater than X.
        # This is an optimization to avoid unnecessary calculations.
        if a > X or b > X:
            return X + 1
        
        # Calculate LCM: (a * b) // gcd(a, b)
        # To prevent intermediate products from becoming unnecessarily large,
        # we calculate a // gcd(a, b) first.
        common_divisor = self._gcd(a, b)
        val_a_prime = a // common_divisor # This is a / gcd(a,b)
        
        # Check if val_a_prime * b would exceed X.
        # This is done by checking if val_a_prime > X / b.
        # Use integer division X // b for the check to avoid floating point issues.
        # If val_a_prime > X // b, it means val_a_prime * b will be > X.
        if val_a_prime > X // b:
            return X + 1 # Indicate that LCM is effectively too large for X
        
        # If it doesn't exceed X, calculate the LCM.
        res_lcm = val_a_prime * b
        return res_lcm

    # Helper function to count distinct amounts <= X using Inclusion-Exclusion Principle
    def _count_amounts_le_X(self, X: int, coins: List[int]) -> int:
        n = len(coins)
        total_distinct_amounts = 0

        # Iterate through all non-empty subsets using bitmask
        # 'i' represents the bitmask for the current subset (from 1 to 2^n - 1)
        for i in range(1, 1 << n):
            current_lcm = 1 # Initialize LCM for the current subset
            subset_size = 0   # Count of elements in the current subset
            
            # Calculate LCM for the current subset
            for j in range(n):
                if (i >> j) & 1: # Check if the j-th coin is in the current subset
                    subset_size += 1
                    # Update current_lcm, capping it at X + 1 if it exceeds X
                    current_lcm = self._safe_lcm(current_lcm, coins[j], X)
                    if current_lcm > X: # If LCM already exceeds X, no multiples are <= X
                        break # No need to continue calculating for this subset, it contributes 0
            
            # If the calculated LCM for the subset is greater than X, it yields 0 amounts.
            if current_lcm > X:
                num_multiples = 0
            else:
                # Number of multiples of current_lcm less than or equal to X
                num_multiples = X // current_lcm
            
            # Apply Inclusion-Exclusion Principle
            # Add for odd-sized subsets, subtract for even-sized subsets
            if subset_size % 2 == 1:
                total_distinct_amounts += num_multiples
            else:
                total_distinct_amounts -= num_multiples
            
        return total_distinct_amounts

    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Binary search for the k-th smallest amount
        
        # Smallest possible amount is 1.
        left = 1 
        
        # The largest possible amount is bounded by max_coin * k.
        # Max coins[i] is 25, max k is 2 * 10^9.
        # So, right = 25 * (2 * 10**9) = 5 * 10**10.
        right = 5 * 10**10 
        
        ans = right # Initialize answer with a value that is certainly an upper bound

        while left <= right:
            mid = left + (right - left) // 2
            
            # Count how many distinct amounts are less than or equal to `mid`
            num_amounts_le_mid = self._count_amounts_le_X(mid, coins)
            
            if num_amounts_le_mid >= k:
                # If we have k or more amounts up to `mid`, then `mid` could be our answer
                # or the actual k-th smallest amount is smaller. Store `mid` and try to find a smaller one.
                ans = mid
                right = mid - 1
            else:
                # If we have fewer than k amounts up to `mid`, then `mid` is too small.
                # The k-th smallest amount must be larger.
                left = mid + 1
                
        return ans