from math import gcd

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        while True:
            # Check if the array is non-decreasing
            if all(nums[i] <= nums[i+1] for i in range(n-1)):
                return operations
            
            # Find the minimum element in the array
            min_val = min(nums)
            
            # Check if it's possible to make the array non-decreasing
            if any(num % min_val == 0 for num in nums):
                return -1
            
            # Perform the operation on the minimum element
            for i in range(n):
                if nums[i] == min_val:
                    nums[i] //= gcd(nums[i], min_val)
                    break
            
            operations += 1