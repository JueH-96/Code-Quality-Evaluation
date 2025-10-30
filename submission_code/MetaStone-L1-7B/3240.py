class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(i_0based, num):
            if i_0based == 0:
                return num
            two_power_i = 1 << i_0based
            two_power_i_plus_1 = two_power_i << 1
            full_cycles = (num + 1) // two_power_i_plus_1
            count = full_cycles * two_power_i
            remainder = (num + 1) % two_power_i_plus_1
            count += max(0, remainder - two_power_i)
            return count
        
        def compute_sum(num):
            max_bit = num.bit_length()
            total = 0
            i = x
            while i <= max_bit:
                i_0based = i - 1
                total += count_set_bits(i_0based, num)
                i += x
            return total
        
        low = 1
        high = 1
        
        while compute_sum(high) <= k:
            high *= 2
        
        while low < high:
            mid = (low + high + 1) // 2
            s = compute_sum(mid)
            if s <= k:
                low = mid
            else:
                high = mid - 1
        
        return low