class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_pairs = n // 2
        
        # Initialize frequency arrays for d and m
        max_d = k
        max_m = k
        freq_d = [0] * (max_d + 1)
        freq_m = [0] * (max_m + 1)
        
        for i in range(total_pairs):
            a = nums[i]
            b = nums[n - 1 - i]
            d = abs(a - b)
            m = max(a, k - a, b, k - b)
            
            freq_d[d] += 1
            freq_m[m] += 1
        
        # Compute suffix sum for m frequencies
        suffix_sum = [0] * (max_m + 2)  # +2 to handle X+1 when X is max_m
        for X in range(max_m, -1, -1):
            suffix_sum[X] = suffix_sum[X + 1] + freq_m[X]
        
        # Find the maximum value of (suffix_sum[X] + freq_d[X])
        max_current = 0
        for X in range(max_m + 1):
            current = suffix_sum[X] + (freq_d[X] if X <= max_d else 0)
            if current > max_current:
                max_current = current
        
        # Calculate minimal cost
        minimal_cost = 2 * total_pairs - max_current
        return minimal_cost