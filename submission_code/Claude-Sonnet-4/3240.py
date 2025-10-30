class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_price_sum(num):
            if num <= 0:
                return 0
            
            total = 0
            bit_pos = 1
            
            while (1 << (bit_pos - 1)) <= num:
                if bit_pos % x == 0:
                    # Count how many numbers from 1 to num have bit set at position bit_pos
                    cycle_length = 1 << bit_pos
                    complete_cycles = (num + 1) // cycle_length
                    remainder = (num + 1) % cycle_length
                    
                    # In each complete cycle, half the numbers have this bit set
                    count = complete_cycles * (cycle_length // 2)
                    
                    # In the incomplete cycle, count additional set bits
                    if remainder > cycle_length // 2:
                        count += remainder - cycle_length // 2
                    
                    total += count
                
                bit_pos += 1
            
            return total
        
        # Binary search for the answer
        left, right = 1, 2 * k  # Upper bound estimation
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if count_price_sum(mid) <= k:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result