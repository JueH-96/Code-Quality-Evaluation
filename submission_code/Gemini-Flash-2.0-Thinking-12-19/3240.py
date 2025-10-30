class Solution:
    def get_price_sum(self, N: int, x: int) -> int:
        total_price_sum = 0
        for m in range(1, 61 // x + 1):
            j = m * x
            power_of_2_j = 1 << j
            power_of_2_j_minus_1 = 1 << (j - 1)
            full_blocks_count = N // power_of_2_j
            remainder = N % power_of_2_j
            count_j_bit_set = full_blocks_count * power_of_2_j_minus_1 + max(0, remainder - power_of_2_j_minus_1 + 1)
            total_price_sum += count_j_bit_set
        return total_price_sum
        
    def findMaximumNumber(self, k: int, x: int) -> int:
        low = 0
        high = 2**60
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            price_sum = self.get_price_sum(mid, x)
            if price_sum <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans