class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        operations = 0
        n = len(nums)
        
        # Calculate the difference between nums and target
        diff = [target[i] - nums[i] for i in range(n)]
        
        # We will count the number of operations needed
        i = 0
        while i < n:
            if diff[i] != 0:
                # We found a non-zero difference, start a segment
                operations += abs(diff[i])
                # Move i to the end of this segment
                while i < n and diff[i] != 0:
                    i += 1
            else:
                i += 1
        
        return operations