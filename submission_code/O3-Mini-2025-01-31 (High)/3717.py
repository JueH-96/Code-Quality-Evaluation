from typing import List
import heapq

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        # m is the number of contiguous subarrays (windows) of length x.
        m = n - x + 1

        # We'll precompute the cost to “fix” each window into a constant subarray.
        # To do that efficiently we need to compute the median and the sum of absolute
        # differences quickly for each sliding window of length x.
        #
        # The cost to transform a window into all equal numbers is minimized by choosing
        # the median as the target. For a window W of length x with median M,
        # cost = sum(|W[i] - M|) = M*(# elements in left) - (sum of left) + (sum of right) - M*(# elements in right)
        #
        # We maintain the sliding window via a two-heaps data structure (left and right heaps)
        # that supports O(log(x)) add and lazy deletion for removals.
        #
        # left is a max-heap (we store negatives) and right is a min-heap.
        # We also maintain:
        #   left_size, right_size: the current number of valid elements in each heap.
        #   left_sum, right_sum: the sum of those valid elements.
        # For lazy removals, we use dictionaries lazy_left and lazy_right keyed by the
        # stored value in each heap.
        
        left = []  # max heap (stores negative numbers)
        right = [] # min heap
        lazy_left = {}
        lazy_right = {}
        left_size = 0
        right_size = 0
        left_sum = 0
        right_sum = 0

        # Clean the top of a heap if its top element has been flagged for removal.
        def prune_heap(heap, lazy):
            while heap and lazy.get(heap[0], 0) > 0:
                top = heapq.heappop(heap)
                lazy[top] -= 1
                if lazy[top] == 0:
                    del lazy[top]

        # Rebalance the heaps so that:
        #   left_size == right_size   OR   left_size == right_size + 1.
        def rebalance():
            nonlocal left_size, right_size, left_sum, right_sum
            prune_heap(left, lazy_left)
            prune_heap(right, lazy_right)
            while left_size > right_size + 1:
                prune_heap(left, lazy_left)
                if not left: break
                val = -heapq.heappop(left)
                left_sum -= val
                left_size -= 1
                heapq.heappush(right, val)
                right_sum += val
                right_size += 1
                prune_heap(left, lazy_left)
                prune_heap(right, lazy_right)
            while right_size > left_size:
                prune_heap(right, lazy_right)
                if not right: break
                val = heapq.heappop(right)
                right_sum -= val
                right_size -= 1
                heapq.heappush(left, -val)
                left_sum += val
                left_size += 1
                prune_heap(left, lazy_left)
                prune_heap(right, lazy_right)

        # Add a new number into our sliding window.
        def add_val(num: int):
            nonlocal left_size, right_size, left_sum, right_sum
            prune_heap(left, lazy_left)
            if left_size == 0 or num <= -left[0]:
                heapq.heappush(left, -num)
                left_sum += num
                left_size += 1
            else:
                heapq.heappush(right, num)
                right_sum += num
                right_size += 1
            rebalance()

        # Remove a number from our sliding window using lazy deletion.
        def remove_val(num: int):
            nonlocal left_size, right_size, left_sum, right_sum
            prune_heap(left, lazy_left)
            # We decide based on the current median (top of left)
            if left_size > 0 and num <= -left[0]:
                key = -num  # because values in left are stored as negatives.
                lazy_left[key] = lazy_left.get(key, 0) + 1
                left_sum -= num
                left_size -= 1
            else:
                lazy_right[num] = lazy_right.get(num, 0) + 1
                right_sum -= num
                right_size -= 1
            prune_heap(left, lazy_left)
            prune_heap(right, lazy_right)
            rebalance()

        # Build the first sliding window of size x.
        for i in range(x):
            add_val(nums[i])
        costSegments = []
        prune_heap(left, lazy_left)
        median = -left[0] if left else 0
        # Compute cost for current window.
        cost = median * left_size - left_sum + right_sum - median * right_size
        costSegments.append(cost)

        # Slide the window: for each new window compute its cost.
        for i in range(1, m):
            remove_val(nums[i - 1])
            add_val(nums[i + x - 1])
            prune_heap(left, lazy_left)
            median = -left[0] if left else 0
            cost = median * left_size - left_sum + right_sum - median * right_size
            costSegments.append(cost)

        # Now, we need to choose at least k non‐overlapping segments (windows)
        # such that the sum of costSegments for them is minimized.
        # A chosen segment starting at index i covers window [i, i+x-1].
        # Non-overlap means that if we choose candidate i, the next candidate must start at i+x or later.
        #
        # We use dynamic programming.
        # Let dp[i][j] be the minimal cost to pick j segments from candidates i .. m-1.
        INF = float('inf')
        dp = [[0] * (k + 1) for _ in range(m + 1)]
        for j in range(1, k + 1):
            dp[m][j] = INF
        dp[m][0] = 0

        # Fill dp table backwards.
        for i in range(m - 1, -1, -1):
            dp[i][0] = 0
            for j in range(1, k + 1):
                # Option 1: skip candidate at i.
                option1 = dp[i + 1][j]
                # Option 2: take candidate at i and jump i+x.
                next_index = i + x
                if next_index > m:
                    next_index = m
                option2 = costSegments[i] + dp[next_index][j - 1]
                dp[i][j] = option1 if option1 < option2 else option2

        return dp[0][k]