class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        result = 0
        window = {}
        
        for right in range(n):
            window[nums[right]] = window.get(nums[right], 0) + 1
            
            while window and max(window.keys()) - min(window.keys()) > 2:
                window[nums[left]] -= 1
                if window[nums[left]] == 0:
                    del window[nums[left]]
                left += 1
            
            result += right - left + 1
            
        return result