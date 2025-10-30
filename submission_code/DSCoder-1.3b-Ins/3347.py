class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = []
        arr2 = []
        result = []
        for i in range(len(nums)):
            if len(arr1) == 0 or nums[i] > arr1[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
            if len(arr1) > 0 and len(arr2) > 0:
                result.append(arr1[-1])
                result.append(arr2[-1])
                arr1 = []
                arr2 = []
        return result