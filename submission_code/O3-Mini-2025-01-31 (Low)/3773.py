from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        def is_non_decreasing(arr: List[int]) -> bool:
            # Check that each element is >= previous element.
            return all(arr[i] >= arr[i-1] for i in range(1, len(arr)))
        
        operations = 0
        
        # Continue operations until array becomes non-decreasing or until we no longer can perform any operation.
        while len(nums) > 1 and not is_non_decreasing(nums):
            # Find the adjacent pair with the minimal sum.
            min_sum = float('inf')
            index = 0  # index of the pair to merge (left index of the pair)
            for i in range(len(nums)-1):
                current_sum = nums[i] + nums[i+1]
                if current_sum < min_sum:
                    min_sum = current_sum
                    index = i
            # Replace the adjacent pair with their sum.
            # Remove the two elements at index and index+1, and insert their sum at index.
            # Using slicing: nums = nums[:index] + [min_sum] + nums[index+2:]
            nums = nums[:index] + [min_sum] + nums[index+2:]
            operations += 1
            
        return operations

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumPairRemoval([5,2,3,1]))  # Expected output: 2
    print(sol.minimumPairRemoval([1,2,2]))    # Expected output: 0