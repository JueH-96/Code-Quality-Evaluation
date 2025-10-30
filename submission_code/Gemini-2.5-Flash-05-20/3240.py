class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        
        # Function to calculate the sum of prices for numbers from 1 to N
        # x_val corresponds to the input 'x'
        def calculate_total_price_sum(N: int, x_val: int) -> int:
            total_price_sum = 0
            
            # Iterate through possible 0-indexed bit positions.
            # The maximum N can be around 2 * 10^15, which is less than 2^51.
            # So, the highest 0-indexed bit that can be set in N is 50.
            # Iterating up to bit_idx = 60 (i.e., considering bits up to 2^60, 1-indexed position 61)
            # is a safe upper bound, covering numbers up to 2^61 - 1.
            for bit_idx in range(61): # Covers 0-indexed bit positions from 0 to 60.
                                     # This means 1-indexed positions from 1 to 61.
                
                # Check if this bit position (1-indexed: bit_idx + 1) is a multiple of x_val.
                if (bit_idx + 1) % x_val == 0:
                    # Calculate countSetBits(N, bit_idx)
                    # This counts how many numbers from 1 to N have the bit_idx-th (0-indexed) bit set.
                    
                    # P = 2^(bit_idx + 1) is the size of a full cycle for this bit
                    p = 1 << (bit_idx + 1)
                    
                    # Count for full blocks: (N // P) full blocks, each contributing (P // 2) set bits
                    count = (N // p) * (p // 2)
                    
                    # Count for the partial block: N % P numbers remaining
                    remainder = N % p
                    # If remainder is >= 2^bit_idx, then the bit is set for 
                    # (remainder - 2^bit_idx + 1) numbers in the partial block.
                    # 2^bit_idx is equivalent to (P // 2)
                    count += max(0, remainder - (p // 2) + 1)
                    
                    total_price_sum += count
            return total_price_sum

        # Binary search for the greatest num
        low = 1
        # A safe upper bound for num.
        # Max k is 10^15.
        # If x=1, price is popcount. Sum of popcounts ~ N * logN / 2.
        # If N = 4 * 10^13, sum ~ 4e13 * 45 / 2 = 9e14. So N can be smaller than k.
        # If x=8, bits are sparse. Sum ~ N * logN / (2 * x).
        # If N = 10^15, sum ~ 1e15 * 50 / 16 = 3e15. So N can be around k.
        # A maximum N around 2 * 10^15 (roughly 2^51) is a safe upper bound.
        high = 2 * (10**15) 
        ans = 0 # Stores the maximum number found so far that satisfies the condition

        while low <= high:
            mid = low + (high - low) // 2
            
            # mid must be at least 1 for price calculation.
            # If low starts at 1, mid will always be >= 1.
            
            current_sum = calculate_total_price_sum(mid, x)
            
            if current_sum <= k:
                # mid is a possible answer, try to find a larger num
                ans = mid
                low = mid + 1
            else:
                # mid is too large, search in the lower half
                high = mid - 1
                
        return ans