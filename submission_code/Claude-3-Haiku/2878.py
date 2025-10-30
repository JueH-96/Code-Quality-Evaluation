class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # Check if all elements are already 0
        if all(num == 0 for num in nums):
            return True
        
        # Iterate through the array and apply the operation
        for i in range(n):
            if nums[i] > 0:
                # Find the minimum element in the subarray of size k
                min_val = min(nums[i:i+k])
                
                # Decrease all elements in the subarray by the minimum value
                for j in range(i, i+k):
                    nums[j] -= min_val
                
                # Check if all elements are now 0
                if all(num == 0 for num in nums):
                    return True
        
        # If we reach this point, it means we couldn't make all elements 0
        return False