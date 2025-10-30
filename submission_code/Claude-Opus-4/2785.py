class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find positions of 1 and n
        pos_1 = nums.index(1)
        pos_n = nums.index(n)
        
        # If already semi-ordered
        if pos_1 == 0 and pos_n == n - 1:
            return 0
        
        # Calculate operations needed
        # To move 1 to position 0, we need pos_1 swaps
        # To move n to position n-1, we need (n-1-pos_n) swaps
        operations = pos_1 + (n - 1 - pos_n)
        
        # If 1 is to the right of n, they will cross during swapping
        # When they cross, we save one operation
        if pos_1 > pos_n:
            operations -= 1
            
        return operations