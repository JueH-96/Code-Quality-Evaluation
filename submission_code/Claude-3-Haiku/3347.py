class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = []
        arr2 = []
        
        # Append the first two elements to the respective arrays
        arr1.append(nums[0])
        arr2.append(nums[1])
        
        # Distribute the remaining elements
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
        
        # Concatenate the two arrays
        result = arr1 + arr2
        
        return result