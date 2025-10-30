class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        total_count = 0
        
        # Check all possible subarrays
        for start in range(n):
            for end in range(start, n):
                # Create a new array without the subarray nums[start:end + 1]
                new_array = nums[:start] + nums[end + 1:]
                
                # Check if the new array is strictly increasing
                if all(new_array[i] < new_array[i + 1] for i in range(len(new_array) - 1)):
                    total_count += 1
        
        return total_count