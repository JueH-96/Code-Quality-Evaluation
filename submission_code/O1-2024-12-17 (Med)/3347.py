class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = [nums[0]]  # first operation: append nums[1] (nums[0] in 0-based index) to arr1
        arr2 = [nums[1]]  # second operation: append nums[2] (nums[1] in 0-based index) to arr2
        
        # subsequent operations
        for i in range(2, len(nums)):  # i = 2 is nums[3] in 1-based index
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
        
        return arr1 + arr2