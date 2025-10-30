import bisect

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        powers = [1]
        while powers[-1] <= 10**9:
            next_power = powers[-1] * 4
            powers.append(next_power)
        
        total_ops = 0
        for l, r in queries:
            k_min = bisect.bisect_right(powers, l)
            k_max = bisect.bisect_right(powers, r)
            
            total_work = 0
            for k in range(k_min, k_max + 1):
                low_bound = powers[k-1]
                high_bound = powers[k] - 1
                seg_low = max(l, low_bound)
                seg_high = min(r, high_bound)
                if seg_low <= seg_high:
                    count = seg_high - seg_low + 1
                    total_work += count * k
            
            op_needed = max((total_work + 1) // 2, k_max)
            total_ops += op_needed
        
        return total_ops