class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_value = 0
        for i in range(1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(nums[j])
            
            if len(subsequence) == 2 * k:
                or_first_half = 0
                for j in range(k):
                    or_first_half |= subsequence[j]
                
                or_second_half = 0
                for j in range(k, 2 * k):
                    or_second_half |= subsequence[j]
                
                max_value = max(max_value, or_first_half ^ or_second_half)
        
        return max_value