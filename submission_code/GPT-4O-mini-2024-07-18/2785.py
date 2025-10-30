class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        first_pos = nums.index(1)
        last_pos = nums.index(n)
        
        # Calculate the number of swaps needed to move 1 to the front
        swaps = first_pos
        
        # Calculate the number of swaps needed to move n to the back
        # If n is before 1 in the list, we need to subtract 1 from the last position
        if last_pos > first_pos:
            swaps += (n - 1 - last_pos)
        else:
            swaps += (n - 1 - last_pos - 1)
        
        return swaps