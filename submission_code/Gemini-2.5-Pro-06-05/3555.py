from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        # We need to perform 'k' operations on the 'nums' list.
        # A for loop that iterates 'k' times is a direct way to model this.
        # The constraints (k <= 10, nums.length <= 100) are small, so a
        # straightforward simulation of the process is efficient enough.
        
        for _ in range(k):
            # In each operation, we need to:
            # 1. Find the minimum value in the current state of 'nums'.
            # 2. If there are duplicates of the minimum value, select the one
            #    that appears first (i.e., has the smallest index).
            # 3. Replace this value with itself multiplied by 'multiplier'.
            
            # The built-in `min()` function can find the minimum value.
            # This is an O(N) operation, where N is the length of `nums`.
            min_value = min(nums)
            
            # The `list.index(value)` method returns the index of the *first*
            # occurrence of the specified value. This perfectly matches the
            # problem's tie-breaking rule. This is also an O(N) operation.
            index_of_min = nums.index(min_value)
            
            # Now, we update the element at the identified index.
            # The list is modified in-place.
            nums[index_of_min] *= multiplier
            
        # After the loop completes, 'nums' holds the final state after 'k' operations.
        return nums