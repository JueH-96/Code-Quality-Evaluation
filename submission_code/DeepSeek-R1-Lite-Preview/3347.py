class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1 = []
        arr2 = []
        
        # Operation 1: append nums[1] to arr1 (nums[0] in Python indexing)
        arr1.append(nums[0])
        
        # Operation 2: append nums[2] to arr2 (nums[1] in Python indexing)
        arr2.append(nums[1])
        
        # Operations from i=3 to n (Python indices 2 to n-1)
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
                
        # Concatenate arr1 and arr2 to form the result
        result = arr1 + arr2
        return result