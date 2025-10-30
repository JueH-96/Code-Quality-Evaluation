from typing import List
import heapq

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        """
        We have an initial list of "segment sums" (each num is its own segment).
        We may merge any two adjacent segments into one whose sum is the sum of the two.
        Our goal is to stop when the list of segment‐sums becomes non‐decreasing, and
        to have as many segments left as possible.  Heuristically we always merge the
        adjacent pair whose combined sum is minimal (to keep new segments small),
        which turns out to maximize the final segment‐count in practice.
        """
        n = len(nums)
        if n <= 1:
            return n

        # Linked‐list style neighbors
        left = list(range(-1, n-1))
        right = list(range(1, n+1))
        # Current sums of segments; at start each is nums[i].
        seg_sum = nums[:]  
        # A min‐heap of (combined_sum, index_of_left_segment, version)
        # We keep a "version" array to skip stale heap entries after merges.
        version = [0]*n
        heap = []
        
        def push_pair(i):
            """If i and right[i] are both valid segments, push their merge‐cost."""
            j = right[i]
            if j < 0 or j >= n:
                return
            heapq.heappush(heap, (seg_sum[i] + seg_sum[j], i, version[i], version[j]))

        # Initialize heap with all adjacent pairs
        for i in range(n-1):
            push_pair(i)

        # Number of segments left
        segments = n
        
        def is_non_decreasing():
            """Check if current seg_sum along the linked list is non-decreasing."""
            i = right[-1]  # head
            prev = seg_sum[i]
            i = right[i]
            while i < n:
                if seg_sum[i] < prev:
                    return False
                prev = seg_sum[i]
                i = right[i]
            return True
        
        # Merge until non-decreasing or only 1 segment remains
        while segments > 1 and not is_non_decreasing():
            total, i, vi, vj = heapq.heappop(heap)
            # Check if i and its right‐neighbor j are still current and unmerged
            if version[i] != vi:
                continue
            j = right[i]
            if j<0 or j>=n or version[j] != vj:
                continue
            # Merge i and j into i
            seg_sum[i] = total
            version[i] += 1
            # Unlink j
            rj = right[j]
            right[i] = rj
            if rj < n:
                left[rj] = i
            # Invalidate j
            version[j] += 1
            segments -= 1
            # Now i may form new merge pairs with its neighbors
            # left neighbor
            pi = left[i]
            if pi >= 0:
                push_pair(pi)
            # right neighbor
            push_pair(i)

        return segments