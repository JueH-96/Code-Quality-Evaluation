from sortedcontainers import SortedList

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        result = 0
        window = SortedList()
        left = 0
        
        for right, value in enumerate(nums):
            window.add(value)
            
            while window[-1] - window[0] > 2:
                window.remove(nums[left])
                left += 1
            
            result += right - left + 1
        
        return result