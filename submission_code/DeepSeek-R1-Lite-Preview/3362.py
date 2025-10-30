from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total = n * (n + 1) // 2
        m = total // 2  # Zero-based index for the median

        # Function to count subarrays with at most k distinct elements
        def count_at_most_k(k):
            freq = defaultdict(int)
            left = 0
            distinct = 0
            count = 0
            for right in range(n):
                if freq[nums[right]] == 0:
                    distinct += 1
                freq[nums[right]] += 1
                # Shrink window from the left if there are more than k distinct elements
                while distinct > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                # Add the number of subarrays ending at right with at most k distinct elements
                count += right - left + 1
            return count

        # Find the maximum possible k
        max_k = len(set(nums))
        # Binary search to find the smallest k such that count_at_most_k >= m
        low = 1
        high = max_k
        while low < high:
            mid = (low + high) // 2
            count = count_at_most_k(mid)
            if count >= m:
                high = mid
            else:
                low = mid + 1
        return low