from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        """
        We want the median of the multiset of 'number of distinct elements' over all subarrays of nums.

        Let n = len(nums).
        The total number of subarrays is M = n*(n+1)//2.
        We define midIndex = (M+1)//2 (1-based). We want the smallest d such that at most-d-distinct subarrays
        count is >= midIndex. That d is our median.

        We use a binary search on d from 1..distinctAll, where distinctAll is the total number of distinct
        elements in nums. For each candidate d, we count the number of subarrays that have at most d distinct
        elements. If that count is >= midIndex, we move to the left; otherwise, we move right.
        The final d we converge on is our median by the problem's definition.
        """

        # Fast helper to count the number of subarrays with at most K distinct elements.
        def count_at_most_k_distinct(arr: List[int], K: int) -> int:
            # If K <= 0, no subarray is possible
            if K <= 0:
                return 0
            n = len(arr)
            freq = [0] * (max_val + 1)
            left = 0
            distinct_count = 0
            total = 0
            for right in range(n):
                x = arr[right]
                if freq[x] == 0:
                    distinct_count += 1
                freq[x] += 1
                while distinct_count > K:
                    y = arr[left]
                    freq[y] -= 1
                    if freq[y] == 0:
                        distinct_count -= 1
                    left += 1
                total += (right - left + 1)
            return total

        n = len(nums)
        if n == 1:
            return 1  # Only one subarray, it has 1 distinct element.

        # Calculate the total number of subarrays M and the target index (1-based) for median
        M = n * (n + 1) // 2
        midIndex = (M + 1) // 2

        # We only need to search up to the total distinct count in the entire array
        distinctAll = len(set(nums))
        max_val = max(nums)  # for sizing the frequency array in our helper

        # If the array has only 1 distinct element overall, the median must be 1
        if distinctAll == 1:
            return 1

        # Binary search on d from 1..distinctAll
        left, right = 1, distinctAll
        while left < right:
            mid = (left + right) // 2
            # subarrays with <= mid distinct
            val = count_at_most_k_distinct(nums, mid)
            if val >= midIndex:
                right = mid
            else:
                left = mid + 1

        return left