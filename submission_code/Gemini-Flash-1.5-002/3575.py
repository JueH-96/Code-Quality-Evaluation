class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_val = 0
        
        for i in range(1 << n):
            subset = []
            for j in range(n):
                if (i >> j) & 1:
                    subset.append(nums[j])
            
            if len(subset) == 2 * k:
                first_half_or = 0
                second_half_or = 0
                for l in range(k):
                    first_half_or |= subset[l]
                for l in range(k, 2*k):
                    second_half_or |= subset[l]
                max_val = max(max_val, first_half_or ^ second_half_or)
        return max_val