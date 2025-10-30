class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                # If we find an element greater than nums[i],
                # nums[i] can't be the maximum for any subarray
                # starting at i and ending at or after j
                if nums[j] > nums[i]:
                    break
                # If nums[j] equals nums[i], then we have a valid subarray
                # where first = last = maximum (which is nums[i])
                if nums[j] == nums[i]:
                    count += 1
        
        return count