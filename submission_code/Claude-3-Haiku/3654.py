class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Sort the array in ascending order
        nums.sort()
        
        # Apply Operation 1 (divide by 2) as many times as possible
        for i in range(op1):
            # Find the largest element that can be divided by 2
            max_idx = next((i for i in range(len(nums)-1, -1, -1) if nums[i] >= 2), -1)
            if max_idx == -1:
                break
            nums[max_idx] = (nums[max_idx] + 1) // 2
        
        # Apply Operation 2 (subtract k) as many times as possible
        for i in range(op2):
            # Find the largest element that is greater than or equal to k
            max_idx = next((i for i in range(len(nums)-1, -1, -1) if nums[i] >= k), -1)
            if max_idx == -1:
                break
            nums[max_idx] -= k
        
        # Return the minimum possible sum of all elements in nums
        return sum(nums)