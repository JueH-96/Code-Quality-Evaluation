class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        for num in nums:
            remainder = num % 3
            if remainder == 1:
                operations += 1
            elif remainder == 2:
                operations += 1
        
        count1 = 0
        count2 = 0
        for num in nums:
            if num % 3 == 1:
                count1 += 1
            elif num % 3 == 2:
                count2 += 1
        
        diff = abs(count1 - count2)
        
        if diff % 3 == 0:
            return operations
        else:
            return operations