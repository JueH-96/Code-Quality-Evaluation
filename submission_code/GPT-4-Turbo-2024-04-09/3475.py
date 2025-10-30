class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1
        
        # Count the number of zeros in the array
        zero_count = nums.count(0)
        
        # If there are no zeros, no operation is needed
        if zero_count == 0:
            return 0
        
        # If the number of zeros is less than 3, it's impossible to flip them all to 1
        if zero_count < 3:
            return -1
        
        operations = 0
        i = 0
        # Traverse the array and flip triples starting from the first zero found
        while i <= n - 3:
            if nums[i] == 0:
                # Flip the next three elements
                nums[i] = 1
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
                operations += 1
            i += 1
        
        # Check if all elements are 1 after operations
        if all(x == 1 for x in nums):
            return operations
        else:
            return -1