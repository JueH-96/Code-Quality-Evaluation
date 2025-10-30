class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        from collections import deque
        # We will maintain a list of segments where each segment is non-decreasing
        segments = []
        i = 0
        while i < n:
            start = i
            while i + 1 < n and nums[i + 1] >= nums[i]:
                i += 1
            segments.append((start, i))
            i += 1
        # The maximum length we can achieve is the sum of lengths of all segments minus the minimal number of merges needed
        # To minimize merges, we need to try to merge minimal number of elements to fix decreases between segments
        # The maximum length is n - (number of merges)
        # Number of merges is (number of segments -1)
        max_length = n - (len(segments) - 1)
        return max_length