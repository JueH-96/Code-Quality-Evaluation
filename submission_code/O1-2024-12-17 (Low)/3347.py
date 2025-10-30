class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = []
        arr2 = []
        
        # First operation: append nums[1] (which is nums[0] in 0-indexing) to arr1
        arr1.append(nums[0])
        # Second operation: append nums[2] (which is nums[1] in 0-indexing) to arr2
        arr2.append(nums[1])
        
        # For subsequent operations from the 3rd to the nth
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
        
        # The result is the concatenation of arr1 and arr2
        return arr1 + arr2