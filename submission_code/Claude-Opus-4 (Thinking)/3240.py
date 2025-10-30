class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_ones_at_position(num, pos):
            # pos is 1-indexed
            cycle_length = 1 << pos  # 2^pos
            complete_cycles = num // cycle_length
            ones_per_cycle = 1 << (pos - 1)  # 2^(pos-1)
            count = complete_cycles * ones_per_cycle
            
            remainder = num % cycle_length
            if remainder >= ones_per_cycle:
                count += remainder - ones_per_cycle + 1
            
            return count
        
        def sum_prices(num, x):
            total = 0
            pos = x
            while (1 << (pos - 1)) <= num:  # While 2^(pos-1) <= num
                count = count_ones_at_position(num, pos)
                total += count
                pos += x
            return total
        
        left, right = 1, 10**16
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if sum_prices(mid, x) <= k:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result