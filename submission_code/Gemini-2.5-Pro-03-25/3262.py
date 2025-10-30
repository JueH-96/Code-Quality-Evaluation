import math # Not strictly necessary for this problem
import sys # Not strictly necessary for this problem
from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        """
        Calculates the largest possible perimeter of a polygon whose sides can be formed from the given list `nums`.

        A polygon must have at least 3 sides.
        The key condition for forming a polygon with k sides (k >= 3) is that the length of the 
        longest side must be strictly smaller than the sum of the lengths of the other sides. 
        If the sides are sorted a_1 <= a_2 <= ... <= a_k, the condition is a_1 + ... + a_{k-1} > a_k.

        The algorithm uses a greedy approach to find the largest perimeter:
        1. Sort the side lengths in ascending order. This allows us to easily identify the longest side 
           of any potential polygon subset and efficiently check the condition.
        2. Start by considering the largest possible subset, which includes all elements of `nums`. 
           Calculate the total sum (potential perimeter).
        3. Check if this largest subset satisfies the polygon condition: is the sum of all elements except 
           the largest one greater than the largest element?
        4. If the condition holds, the sum of all elements is the largest possible perimeter, so we return it.
        5. If the condition does not hold, it means the largest element (`nums[n-1]`) is too long to form 
           a polygon with the remaining elements. We discard this largest element because any polygon including 
           it as the longest side would also fail the condition.
        6. Update the potential perimeter by subtracting the discarded longest side. Now consider the subset 
           consisting of the remaining `n-1` elements. Repeat the check (step 3) with the new largest element 
           (`nums[n-2]`) and the updated sum of others.
        7. Continue this process, discarding the largest element of the current subset if the polygon condition 
           fails, until either a valid polygon is found (and its perimeter returned) or the number of sides 
           in the considered subset drops below 3.
        8. If the loop finishes without finding a valid polygon (i.e., we run out of subsets with 
           at least 3 sides), it means no polygon can be formed from the given numbers, so we return -1.

        Args:
            nums: A list of positive integers representing potential side lengths.
                 Constraints: 3 <= len(nums) <= 10^5, 1 <= nums[i] <= 10^9.

        Returns:
            The largest possible perimeter of a valid polygon that can be formed using sides from `nums`, 
            or -1 if no such polygon can be formed.
        """
        n = len(nums)

        # Sort the array in ascending order. This step is crucial for the greedy strategy.
        # Time complexity: O(n log n), where n is the number of elements in nums.
        nums.sort()

        # Calculate the sum of all side lengths initially. This represents the perimeter
        # if the largest possible subset (all numbers) forms a polygon.
        # Time complexity: O(n)
        current_perimeter = sum(nums)

        # Iterate downwards, considering subsets of decreasing size.
        # 'i' is the index of the largest element in the current subset nums[0...i].
        # The loop starts with the full set (i = n-1) and goes down to the smallest
        # possible polygon size (3 sides, corresponding to i = 2).
        # The loop condition `i >= 2` ensures we always consider subsets with at least 3 sides (k = i + 1 >= 3).
        # Time complexity of the loop: O(n) in the worst case (if we check all subsets down to size 3).
        for i in range(n - 1, 1, -1): # This loop iterates through i = n-1, n-2, ..., 2
            
            # In the current subset nums[0...i], the longest side is nums[i] because the array is sorted.
            longest_side = nums[i]

            # Calculate the sum of the other sides in the current subset.
            # This is the total perimeter of the current subset minus the longest side.
            sum_of_others = current_perimeter - longest_side

            # Check the polygon validity condition: sum of other sides > longest side.
            if sum_of_others > longest_side:
                # If the condition holds, the subset nums[0...i] forms a valid polygon.
                # Since we are iterating from the largest possible sum downwards,
                # this `current_perimeter` (sum of nums[0...i]) is the largest possible perimeter.
                return current_perimeter
            else:
                # If the condition fails, the subset nums[0...i] cannot form a valid polygon
                # because the longest side nums[i] is too large relative to the sum of the others.
                # We must exclude nums[i]. We update the `current_perimeter` by subtracting
                # the discarded side's length, effectively preparing for the check of the
                # next smaller subset (nums[0...i-1]) in the following iteration.
                current_perimeter -= longest_side # current_perimeter now holds the sum for nums[0...i-1]

        # If the loop completes without returning a value, it means no subset of size 3 or greater
        # satisfied the polygon condition. Therefore, it's impossible to form a polygon.
        return -1