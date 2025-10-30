from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        # Build a frequency array for numbers 0..n-1 (nums[i] < n by constraints)
        freq = [0] * n
        for num in nums:
            freq[num] += 1

        # Build prefix sum: prefix[k] = count of elements with value strictly less than k.
        # Since valid candidate k runs from 0 to n, we need prefix for index n+1.
        prefix = [0] * (n + 1)
        for k in range(1, n + 1):
            prefix[k] = prefix[k - 1] + freq[k - 1]

        ways = 0
        # Check each possible group size k from 0 to n.
        for k in range(0, n + 1):
            # For a valid selection with k students:
            # 1. All students with nums[i] < k must be selected. Their count is prefix[k].
            #    So we must have prefix[k] == k.
            # 2. Also, any student with nums[i] == k cannot be made happy (since whether selected or not, condition fails)
            #    That is, freq[k] should be zero (when k < n). For k == n, nums[i] == n is impossible.
            if prefix[k] == k:
                if k < n:
                    if freq[k] == 0:
                        ways += 1
                else:
                    # When k == n, this condition is automatically satisfied because nums[i] < n
                    ways += 1

        return ways