class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Helper function to check if the array is strictly increasing
        def is_strictly_increasing(arr):
            return all(arr[i] < arr[i + 1] for i in range(len(arr) - 1))
        
        # Check all possible subarrays
        for start in range(n):
            for end in range(start, n):
                # Create the new array after removing the subarray from start to end
                new_array = nums[:start] + nums[end + 1:]
                # Check if the new array is strictly increasing
                if is_strictly_increasing(new_array):
                    count += 1
        
        return count