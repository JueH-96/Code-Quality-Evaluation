from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        # Helper function to check if an array is non-decreasing.
        def is_non_decreasing(arr: List[int]) -> bool:
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False
            return True
        
        if is_non_decreasing(nums):
            return 0
        
        operations = 0
        
        while not is_non_decreasing(nums):
            n = len(nums)
            # Find the adjacent pair with the minimum sum, taking left-most on tie.
            min_sum = float('inf')
            min_index = -1
            for i in range(n - 1):
                s = nums[i] + nums[i+1]
                if s < min_sum:
                    min_sum = s
                    min_index = i
            # Replace the chosen adjacent pair with their sum.
            new_val = nums[min_index] + nums[min_index+1]
            nums = nums[:min_index] + [new_val] + nums[min_index+2:]
            operations += 1
        
        return operations

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumPairRemoval([5,2,3,1]))  # Expected output: 2
    print(sol.minimumPairRemoval([1,2,2]))    # Expected output: 0