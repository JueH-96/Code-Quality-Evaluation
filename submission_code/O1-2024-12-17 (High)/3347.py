class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = []
        arr2 = []
        
        # First operation
        arr1.append(nums[0])
        # Second operation
        arr2.append(nums[1])
        
        # Subsequent operations
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
                
        return arr1 + arr2