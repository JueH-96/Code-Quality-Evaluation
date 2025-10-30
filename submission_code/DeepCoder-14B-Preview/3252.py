class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                # Create the new array after removing the subarray from i to j
                new_arr = nums[:i] + nums[j+1:]
                # Check if the new array is strictly increasing
                is_increasing = True
                for k in range(len(new_arr) - 1):
                    if new_arr[k] >= new_arr[k+1]:
                        is_increasing = False
                        break
                if is_increasing:
                    count += 1
        
        return count