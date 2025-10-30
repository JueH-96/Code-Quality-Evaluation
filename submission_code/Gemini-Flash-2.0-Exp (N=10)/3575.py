class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_val = 0
        for i in range(1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(nums[j])
            
            if len(subsequence) == 2 * k:
                first_half_or = 0
                for j in range(k):
                    first_half_or |= subsequence[j]
                
                second_half_or = 0
                for j in range(k, 2 * k):
                    second_half_or |= subsequence[j]
                
                max_val = max(max_val, first_half_or ^ second_half_or)
        
        return max_val