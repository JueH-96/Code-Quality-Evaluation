class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from sortedcontainers import SortedDict
        
        n = len(nums)
        left = 0
        count = 0
        
        counter = SortedDict()
        
        for right in range(n):
            # Add the current element to our window
            counter[nums[right]] = counter.get(nums[right], 0) + 1
            
            # If the window is invalid (max - min > 2), shrink from the left
            while counter and counter.peekitem(-1)[0] - counter.peekitem(0)[0] > 2:
                counter[nums[left]] -= 1
                if counter[nums[left]] == 0:
                    del counter[nums[left]]
                left += 1
            
            # Count all valid subarrays ending at the current position
            count += right - left + 1
        
        return count