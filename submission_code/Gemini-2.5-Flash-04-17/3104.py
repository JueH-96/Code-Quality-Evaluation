from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        """
        Counts the number of ways to select a group of students so that everyone remains happy.

        Args:
            nums: A list of integers where nums[i] is a threshold for the i-th student.

        Returns:
            The number of ways to select a group of students.
        """
        n = len(nums)
        # Sort the array in non-decreasing order.
        # Let the sorted array be s. s[0] <= s[1] <= ... <= s[n-1].
        nums.sort()

        count = 0

        # We consider the number of selected students, k, which can range from 0 to n.
        # For a fixed number of selected students k to be valid (i.e., make everyone happy),
        # two conditions must hold for all students:
        # 1. If student i is selected, then k > nums[i]. This implies that all selected students must have nums[i] < k.
        # 2. If student i is not selected, then k < nums[i]. This implies that all not-selected students must have nums[i] > k.
        # From these conditions, it follows that if a valid selection of size k exists, the set of selected students must be exactly {i | nums[i] < k}, and the set of not-selected students must be exactly {j | nums[j] > k}.
        # This is only possible if the number of students i with nums[i] < k is precisely k, AND there are no students p with nums[p] == k.

        # In the sorted array s, the condition |{i | nums[i] < k}| == k and "no student has nums[p] == k" translates to checks at the boundary points corresponding to k.

        # Case k = 0 (select 0 students):
        # This corresponds to a "cut" point before the first element (index -1).
        # All n students are not selected. The condition for each student i is:
        # (student i is not selected AND total selected < nums[i]).
        # This means 0 < nums[i] for all i.
        # In the sorted array s, this is equivalent to the smallest element being > 0.
        # i.e., s[0] > 0.
        # If this holds, then |{i | nums[i] < 0}| = 0 (always true since nums[i] >= 0) and 0 is not in nums (since s[0] > 0). So k=0 is a valid way.
        if nums[0] > 0:
            count += 1

        # Case k = 1 to n-1 (select k students):
        # A valid number of selected students k (1 <= k <= n-1) corresponds
        # to a "cut" point between index k-1 and index k in the sorted array.
        # The k students with the smallest nums values (indices 0 to k-1) are selected.
        # The n-k students with the largest nums values (indices k to n-1) are not selected.
        # For the k selected students (with values s[0]...s[k-1]), the condition is k > value.
        # The strongest requirement among these is k > s[k-1], which is s[k-1] < k. (This applies since k >= 1).
        # For the n-k not-selected students (with values s[k]...s[n-1]), the condition is k < value.
        # The strongest requirement among these is k < s[k], which is s[k] > k. (This applies since k <= n-1).
        # If s[k-1] < k and s[k] > k, then precisely k elements are < k (namely s[0]...s[k-1]), and k is not present in the array.
        # So, for 1 <= k <= n-1, a selection of k students is valid iff nums[k-1] < k AND nums[k] > k.
        for k in range(1, n):
            if nums[k - 1] < k and nums[k] > k:
                count += 1

        # Case k = n (select n students):
        # This corresponds to a "cut" point after the last element (index n-1).
        # All n students are selected. The condition for each student i is:
        # (student i is selected AND total selected > nums[i]).
        # This means n > nums[i] for all i.
        # In the sorted array s, this is equivalent to the largest element being < n.
        # i.e., s[n-1] < n.
        # If this holds, then |{i | nums[i] < n}| = n and n is not in nums (since s[n-1] < n and nums[i] <= n-1). So k=n is a valid way.
        if nums[n - 1] < n:
            count += 1

        return count