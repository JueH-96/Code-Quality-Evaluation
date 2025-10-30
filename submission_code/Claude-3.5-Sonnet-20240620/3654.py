class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        
        # Sort the array in descending order
        nums.sort(reverse=True)
        
        # Apply operation 1 (divide by 2) to the largest elements
        for i in range(min(op1, n)):
            nums[i] = (nums[i] + 1) // 2
        
        # Sort again after applying operation 1
        nums.sort(reverse=True)
        
        # Apply operation 2 (subtract k) to the largest elements
        for i in range(min(op2, n)):
            if nums[i] >= k:
                nums[i] -= k
        
        # Return the sum of the resulting array
        return sum(nums)