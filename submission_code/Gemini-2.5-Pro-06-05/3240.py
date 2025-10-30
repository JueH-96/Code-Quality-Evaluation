class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        
        def get_price_sum(num: int) -> int:
            """
            Calculates the sum of prices for all numbers from 1 to num.
            """
            if num == 0:
                return 0
            
            total_price = 0
            # Iterate through bit positions i that are multiples of x.
            # Bit positions are 1-indexed.
            # The maximum number of bits for numbers up to ~10^18 is around 60.
            # A loop up to 63 is safe.
            for i in range(x, 64, x):
                # Calculate how many numbers from 1 to num have the i-th bit set.
                period = 1 << i
                half_period = 1 << (i - 1)
                
                # Consider numbers from 0 to num, which has length num + 1.
                num_plus_1 = num + 1
                
                # Number of full periods
                num_periods = num_plus_1 // period
                count_from_periods = num_periods * half_period
                
                # Contribution from the remaining part of the last, incomplete period
                remainder = num_plus_1 % period
                count_from_remainder = max(0, remainder - half_period)
                
                total_price += count_from_periods + count_from_remainder
            return total_price

        # Binary search for the answer num.
        low = 0
        # A sufficiently large upper bound. A simple heuristic like (k+1) * 2^x works well.
        # k can be up to 10^15, x up to 8. This can be large but is handled by Python's arbitrary-precision integers.
        high = (k + 1) * (1 << x)
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            
            current_sum = get_price_sum(mid)
            
            if current_sum <= k:
                # mid is a potential answer. Try for a larger one.
                ans = mid
                low = mid + 1
            else:
                # mid is too large.
                high = mid - 1
                
        return ans