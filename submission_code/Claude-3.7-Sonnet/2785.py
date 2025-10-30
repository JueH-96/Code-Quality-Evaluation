class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find positions of 1 and n
        pos_1 = nums.index(1)
        pos_n = nums.index(n)
        
        # Calculate operations needed:
        # - pos_1 swaps to move 1 to the beginning
        # - (n-1-pos_n) swaps to move n to the end
        operations = pos_1 + (n - 1 - pos_n)
        
        # If 1 appears after n in the array, when we move 1 to the front,
        # it will also shift n one position to the right, saving us one operation
        if pos_1 > pos_n:
            operations -= 1
            
        return operations