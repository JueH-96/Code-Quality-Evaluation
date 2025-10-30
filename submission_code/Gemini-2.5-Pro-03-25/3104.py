import math # Not used, but often included in competitive programming templates
from typing import List # Import List from typing

class Solution:
    """
    Solves the problem of counting happy student group selections.
    """
    def countWays(self, nums: List[int]) -> int:
        """
        Finds the number of ways to select a group of students such that everyone remains happy.

        A student i is happy if:
        - Selected and the total number of selected students k satisfies k > nums[i].
        - Not selected and the total number of selected students k satisfies k < nums[i].

        We need to find the number of possible values for k (the number of selected students, ranging from 0 to n)
        such that it's possible to form a group of size k where everyone is happy.

        Let's analyze the conditions for a given k to be a valid number of selected students.
        If k students are selected, then:
        1. Every selected student `i` must satisfy `k > nums[i]`.
        2. Every not selected student `j` must satisfy `k < nums[j]`.

        This implies that all students `i` with `nums[i] >= k` must NOT be selected,
        and all students `j` with `nums[j] < k` MUST be selected.
        Crucially, no student `x` can have `nums[x] == k`, because such a student could never be happy.

        If we sort the `nums` array, let it be `nums_sorted`.
        Consider a potential number of selected students `k`.

        If we select the `k` students corresponding to the smallest `nums` values (i.e., `nums_sorted[0]` to `nums_sorted[k-1]`),
        and leave the remaining `n-k` students (corresponding to `nums_sorted[k]` to `nums_sorted[n-1]`) not selected.
        Let's check if this specific selection satisfies the happiness conditions for everyone.

        - For the selected students `i` (indices 0 to `k-1`): We need `k > nums_sorted[i]`. Since the array is sorted, this is equivalent to checking the largest value among them: `k > nums_sorted[k-1]` (if `k > 0`).
        - For the not selected students `j` (indices `k` to `n-1`): We need `k < nums_sorted[j]`. Since the array is sorted, this is equivalent to checking the smallest value among them: `k < nums_sorted[k]` (if `k < n`).

        So, a value `k` represents a valid way if these conditions are met for the selection of the first k students (in sorted order):

        Case 1: k = 0 (no students selected)
        - All n students are not selected.
        - Condition: `k < nums_sorted[j]` for all `j` from 0 to `n-1`.
        - This means `0 < nums_sorted[0]`.

        Case 2: k = n (all students selected)
        - All n students are selected.
        - Condition: `k > nums_sorted[i]` for all `i` from 0 to `n-1`.
        - This means `n > nums_sorted[n-1]`.

        Case 3: 0 < k < n
        - Selected students are indices 0 to `k-1`. Not selected are `k` to `n-1`.
        - Condition 1 (selected): `k > nums_sorted[k-1]`.
        - Condition 2 (not selected): `k < nums_sorted[k]`.
        - Combined: `nums_sorted[k-1] < k < nums_sorted[k]`.

        The algorithm iterates through `k` from 0 to `n` and counts how many values satisfy these conditions.

        Args:
            nums: A list of integers representing the happiness threshold for each student.

        Returns:
            The number of ways (number of possible values for k) to select students happily.
        """
        n = len(nums)
        # Sorting allows efficient checking of conditions based on k
        # O(n log n) time complexity for sorting.
        nums.sort()

        count = 0 # Initialize the count of valid ways (valid k values)

        # Check the case k = 0 (selecting 0 students)
        # Condition: nums[0] > 0 (using the sorted array)
        # This ensures all not selected students j satisfy 0 < nums[j].
        # The constraints state 1 <= n, so nums[0] is always accessible.
        if nums[0] > 0:
            count += 1

        # Check cases for k from 1 to n-1
        # The loop iterates n-1 times. Each check is O(1). Total O(n).
        # Condition: nums[k-1] < k < nums[k] (using the sorted array)
        # This ensures happiness for both selected (first k) and not selected (remaining n-k) students.
        for k in range(1, n):
            # Check if the condition holds for the current k
            if nums[k-1] < k and k < nums[k]:
                count += 1

        # Check the case k = n (selecting all n students)
        # Condition: nums[n-1] < n (using the sorted array)
        # This ensures all selected students i satisfy n > nums[i].
        # The constraints state 1 <= n, so nums[n-1] is always accessible.
        if nums[n-1] < n:
            count += 1

        # The overall time complexity is dominated by sorting: O(n log n).
        # The space complexity is O(n) or O(log n) depending on the sort implementation.
        return count # Return the total count of valid ways