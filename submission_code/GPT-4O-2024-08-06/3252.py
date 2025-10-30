class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Helper function to check if an array is strictly increasing
        def is_strictly_increasing(arr):
            for i in range(1, len(arr)):
                if arr[i] <= arr[i - 1]:
                    return False
            return True
        
        # Iterate over all possible subarrays
        for start in range(n):
            for end in range(start, n):
                # Create a new array by removing the subarray nums[start:end+1]
                new_array = nums[:start] + nums[end+1:]
                # Check if the new array is strictly increasing
                if is_strictly_increasing(new_array):
                    count += 1
        
        return count