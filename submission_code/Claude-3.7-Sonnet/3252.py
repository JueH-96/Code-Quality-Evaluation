class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for start in range(n):
            for end in range(start, n):
                # Check if the array is still strictly increasing after removing subarray [start, end]
                
                # Check if portion before the removed subarray is strictly increasing
                before_increasing = True
                for i in range(1, start):
                    if nums[i] <= nums[i-1]:
                        before_increasing = False
                        break
                
                if not before_increasing:
                    continue
                
                # Check if portion after the removed subarray is strictly increasing
                after_increasing = True
                for i in range(end+2, n):
                    if nums[i] <= nums[i-1]:
                        after_increasing = False
                        break
                
                if not after_increasing:
                    continue
                
                # Check transition between before and after portions
                transition_valid = True
                if start > 0 and end < n-1 and nums[end+1] <= nums[start-1]:
                    transition_valid = False
                
                if transition_valid:
                    count += 1
        
        return count