import math

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        total_operations = 0
        
        for l, r in queries:
            sum_steps = 0
            k = 0
            # The maximum k for numbers up to r is floor(log4(r)) + 1
            max_k = math.floor(math.log(r, 4)) + 2 if r > 0 else 0  # Handle r=0 case, though constraints say l >=1
            
            # We'll process each k where 4^{k-1} <= x < 4^k has step k
            # So for each k, find the numbers in [l, r] that are in [4^{k-1}, 4^k -1]
            k = 1
            current_sum = 0
            while True:
                lower = 4 ** (k - 1)
                upper = (4 ** k) - 1
                if lower > r:
                    break
                # The overlap between [l, r] and [lower, upper]
                overlap_l = max(l, lower)
                overlap_r = min(r, upper)
                if overlap_l <= overlap_r:
                    count = overlap_r - overlap_l + 1
                    current_sum += count * k
                k += 1
            # The minimal operations is ceil(current_sum / 2)
            total_operations += (current_sum + 1) // 2
        
        return total_operations