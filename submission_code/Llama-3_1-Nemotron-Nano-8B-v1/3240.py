class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def compute_sum(num):
            sum_total = 0
            highest_bit = num.bit_length()
            j = x
            while j <= highest_bit:
                cycle = 1 << j  # 2^j
                full_cycles = (num + 1) // cycle
                count = full_cycles * (1 << (j - 1))
                remainder = (num + 1) % cycle
                add = max(0, remainder - (1 << (j - 1)))
                sum_total += count + add
                j += x
            return sum_total
        
        low, high = 1, 1 << 60  # Arbitrary large upper bound
        best = 0
        while low <= high:
            mid = (low + high) // 2
            current_sum = compute_sum(mid)
            if current_sum <= k:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best