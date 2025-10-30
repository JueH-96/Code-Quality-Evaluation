class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        for i in range(n):
            # If current element is not 0, we need to apply operations
            if nums[i] > 0:
                # Check if we have enough elements to form a subarray of size k
                if i + k > n:
                    return False
                
                # Apply nums[i] operations on the subarray starting at i
                operations = nums[i]
                for j in range(i, i + k):
                    nums[j] -= operations
                    if nums[j] < 0:
                        return False
        
        return True