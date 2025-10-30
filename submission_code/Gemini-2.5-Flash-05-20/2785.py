from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        idx_1 = -1
        idx_n = -1
        
        # Find the positions of 1 and n in the permutation.
        # Iterate through the array once to find both indices.
        for i in range(n):
            if nums[i] == 1:
                idx_1 = i
            if nums[i] == n:
                idx_n = i
            # Optimization: if both indices are found, we can break early.
            if idx_1 != -1 and idx_n != -1:
                break
                
        # Calculate the number of swaps required to move 1 to the first position (index 0).
        # This is simply its current index, as each adjacent swap moves it one position closer to 0.
        swaps_to_move_1 = idx_1
        
        # Calculate the number of swaps required to move n to the last position (index n-1).
        # This is the distance from its current index to the last index.
        swaps_to_move_n = (n - 1) - idx_n
        
        # The total number of swaps is initially the sum of swaps for 1 and n
        # if their movements were completely independent.
        total_swaps = swaps_to_move_1 + swaps_to_move_n
        
        # However, if 1 is initially to the right of n (i.e., idx_1 > idx_n),
        # their paths will cross. When 1 moves left, it will eventually swap with n.
        # This particular swap counts for both 1 moving left AND n effectively moving right.
        # Therefore, one operation is "shared" or "saved", and we must subtract 1 from the total.
        if idx_1 > idx_n:
            total_swaps -= 1
            
        return total_swaps