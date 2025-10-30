class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = []
        arr2 = []
        
        # First operation: append nums[0] to arr1
        arr1.append(nums[0])
        
        # Second operation: append nums[1] to arr2
        if len(nums) > 1:
            arr2.append(nums[1])
        
        # Subsequent operations
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
        
        # Concatenate arr1 and arr2
        result = arr1 + arr2
        return result