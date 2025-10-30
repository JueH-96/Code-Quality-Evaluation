from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)

        # Iterate through all possible starting indices for the distinct suffix.
        # 'i' represents the number of elements that would be removed from the beginning of the array.
        # The remaining suffix is nums[i:].
        # We are looking for the smallest 'i' such that nums[i:] has distinct elements.
        # This minimum 'i' will yield the minimum number of operations.
        for i in range(n + 1): 
            # 'i' ranges from 0 (no elements removed) to n (all elements removed, leaving an empty array).
            current_suffix = nums[i:]

            # Check if the current_suffix has distinct elements.
            # Using a set for efficient average O(1) time lookups and insertions.
            seen = set()
            is_distinct = True
            for num in current_suffix:
                if num in seen:
                    is_distinct = False
                    break # Found a duplicate, this suffix is not distinct.
                seen.add(num)

            if is_distinct:
                # This is the smallest 'i' for which nums[i:] is distinct.
                # Since we are iterating 'i' in increasing order, the first 'i' that
                # satisfies the distinctness condition is the minimum 'i'.
                
                # Calculate the number of operations: ceil(i / 3).
                # Using integer division, ceil(a / b) can be calculated as (a + b - 1) // b.
                # Here, b = 3, so it becomes (i + 3 - 1) // 3, which simplifies to (i + 2) // 3.
                return (i + 2) // 3

        # This part of the code should technically not be reached because when i = n,
        # current_suffix will be an empty list ([]), which is always considered distinct.
        # Thus, a return value is guaranteed for any valid input `nums`.