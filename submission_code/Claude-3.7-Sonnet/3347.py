class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # Initialize the two arrays with the first two elements
        arr1 = [nums[0]]  # First element goes to arr1
        arr2 = [nums[1]]  # Second element goes to arr2
        
        # Process the remaining elements according to the rule
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
        
        # Return the concatenated result
        return arr1 + arr2