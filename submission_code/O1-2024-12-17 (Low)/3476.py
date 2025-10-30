class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        
        for num in nums:
            remainder = num % 3
            if remainder != 0:
                # cost is the minimum of either adding or subtracting (distance to multiple of 3)
                # for remainder 1, cost can be 1 (subtract 1), 
                # for remainder 2, cost can be 1 (add 1), 
                # so either case is 1 operation
                operations += 1
        
        return operations