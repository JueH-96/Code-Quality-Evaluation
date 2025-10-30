class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                # Create the new array by removing the subarray from i to j
                new_nums = nums[:i] + nums[j+1:]
                # Check if the new array is strictly increasing
                is_strictly_increasing = True
                for k in range(1, len(new_nums)):
                    if new_nums[k] <= new_nums[k-1]:
                        is_strictly_increasing = False
                        break
                if is_strictly_increasing:
                    count += 1
        
        return count