class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = [nums[0]]  # First operation: append nums[1] to arr1 (1-indexed, so nums[0])
        arr2 = [nums[1]]  # Second operation: append nums[2] to arr2 (1-indexed, so nums[1])
        
        # Process remaining elements starting from index 2 (3rd element in 1-indexed)
        for i in range(2, len(nums)):
            # Compare last elements of arr1 and arr2
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
        
        # Concatenate arr1 and arr2
        return arr1 + arr2