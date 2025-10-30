class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        max_freq = 1
        
        for i in range(n):
            left, right = i, i
            curr_freq = 1
            
            while right + 1 < n and nums[right + 1] == nums[i]:
                right += 1
                curr_freq += 1
            
            if curr_freq > max_freq:
                max_freq = curr_freq
            
            if k > 0:
                while left > 0 and nums[left - 1] == nums[i]:
                    left -= 1
                    k -= 1
                    curr_freq += 1
                    
                    if curr_freq > max_freq:
                        max_freq = curr_freq
        
        return max_freq