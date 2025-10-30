class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        score = 0
        i = 0
        
        while i < n - 1:
            # Find the next index with a value greater than nums[i]
            j = i + 1
            while j < n - 1 and nums[j] <= nums[i]:
                j += 1
            
            # Jump to position j (or to the end if no better position found)
            score += (j - i) * nums[i]
            i = j
        
        return score