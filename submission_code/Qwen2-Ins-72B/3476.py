class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Count the numbers that are 1 or 2 modulo 3
        mod1 = sum(1 for x in nums if x % 3 == 1)
        mod2 = sum(1 for x in nums if x % 3 == 2)
        
        # Calculate the minimum operations needed
        operations = 0
        while mod1 > 0 or mod2 > 0:
            if mod1 > 0 and mod2 > 0:
                # If there are numbers that are 1 and 2 modulo 3, pair them
                mod1 -= 1
                mod2 -= 1
                operations += 1
            elif mod1 > 0:
                # If there are only numbers that are 1 modulo 3, pair them in groups of 3
                mod1 -= 3
                operations += 2
            elif mod2 > 0:
                # If there are only numbers that are 2 modulo 3, pair them in groups of 3
                mod2 -= 3
                operations += 2
        
        return operations