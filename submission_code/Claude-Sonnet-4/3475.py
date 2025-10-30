class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        # Make a copy to avoid modifying the original array
        arr = nums[:]
        
        for i in range(n - 2):  # We can only flip if there are at least 3 elements left
            if arr[i] == 0:
                # Flip the current element and the next 2 elements
                arr[i] = 1 - arr[i]
                arr[i + 1] = 1 - arr[i + 1]
                arr[i + 2] = 1 - arr[i + 2]
                operations += 1
        
        # Check if the last two elements are 1
        if arr[n - 2] == 0 or arr[n - 1] == 0:
            return -1
        
        return operations