class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits_at_bit_position(num, bit_pos):
            # Count how many numbers from 1 to num have a set bit at bit_pos (0-indexed from right)
            if num <= 0:
                return 0
            
            cycle_length = 1 << (bit_pos + 1)  # 2^(bit_pos+1)
            complete_cycles = (num + 1) // cycle_length
            remainder = (num + 1) % cycle_length
            
            count = complete_cycles * (1 << bit_pos)
            if remainder > (1 << bit_pos):
                count += remainder - (1 << bit_pos)
            
            return count
        
        def total_price_sum(num):
            # Compute sum of prices of all numbers from 1 to num
            total = 0
            bit_pos = x - 1  # Convert 1-indexed position x to 0-indexed bit_pos
            while bit_pos < 60:  # Enough bits to handle large numbers
                total += count_set_bits_at_bit_position(num, bit_pos)
                bit_pos += x
            return total
        
        # Binary search for the answer
        left, right = 1, 10**16
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if total_price_sum(mid) <= k:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result