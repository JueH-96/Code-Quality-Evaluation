class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        low = 1
        high = 1
        
        # Find an upper bound where the sum exceeds k
        while True:
            current_sum = self.compute_sum(high, x)
            if current_sum > k:
                break
            high *= 2
        
        best = 0
        while low <= high:
            mid = (low + high) // 2
            s = self.compute_sum(mid, x)
            if s <= k:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best
    
    def compute_sum(self, num: int, x: int) -> int:
        if num == 0:
            return 0
        sum_count = 0
        m = num.bit_length()
        j_max = (m // x) * x
        for j in range(x, j_max + 1, x):
            cycle = 1 << j      # 2^j
            half = 1 << (j - 1) # 2^(j-1)
            full_cycles = num // cycle
            remainder = num % cycle
            count = full_cycles * half
            if remainder >= half:
                count += remainder - half + 1
            sum_count += count
        return sum_count