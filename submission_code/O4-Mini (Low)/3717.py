import heapq
from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        if k * x > n:
            return -1  # impossible, though by constraints won't happen

        # Helper for sliding window median + cost maintenance
        # We keep two heaps: max-heap 'low' for lower half, min-heap 'high' for upper half
        # Also maintain sums of each half for cost calculation
        low = []   # max-heap (store negatives)
        high = []  # min-heap
        low_sum = 0
        high_sum = 0
        delayed = {}  # for lazy removals

        def prune(heap):
            # Remove the top of heap if it's delayed
            while heap:
                val = heap[0]
                # actual value key
                key = -val if heap is low else val
                if delayed.get(key, 0):
                    # Pop and decrement delayed count
                    heapq.heappop(heap)
                    delayed[key] -= 1
                    if delayed[key] == 0:
                        del delayed[key]
                else:
                    break

        def rebalance():
            nonlocal low_sum, high_sum
            # balance sizes: allow len(low) == len(high) or len(low) == len(high)+1
            if len(low) > len(high) + 1:
                prune(low)
                v = -heapq.heappop(low)
                low_sum -= v
                heapq.heappush(high, v)
                high_sum += v
            elif len(low) < len(high):
                prune(high)
                v = heapq.heappop(high)
                high_sum -= v
                heapq.heappush(low, -v)
                low_sum += v

        def add(num):
            nonlocal low_sum, high_sum
            if not low or num <= -low[0]:
                heapq.heappush(low, -num)
                low_sum += num
            else:
                heapq.heappush(high, num)
                high_sum += num
            rebalance()

        def remove(num):
            nonlocal low_sum, high_sum
            # delay removal
            delayed[num] = delayed.get(num, 0) + 1
            # adjust sums immediately depending on where num would be
            if low and num <= -low[0]:
                low_sum -= num
                # might need prune top later
            else:
                high_sum -= num
            # then rebalance to fix sizes
            prune(low)
            prune(high)
            rebalance()

        def get_cost():
            # median is at top of low
            prune(low)
            m = -low[0]
            # cost = sum{|a - m|} = m*len(low) - low_sum + high_sum - m*len(high)
            return m * len(low) - low_sum + high_sum - m * len(high)

        # Step 1: compute cost for each window of size x
        costs = [0] * (n - x + 1)
        # initialize first window
        for i in range(x):
            add(nums[i])
        costs[0] = get_cost()
        for i in range(x, n):
            add(nums[i])
            remove(nums[i - x])
            costs[i - x + 1] = get_cost()

        # Step 2: DP to pick k non-overlapping windows of length x with min sum cost
        INF = 10**30
        # dp[j][i] = min cost to pick j windows among indices >= i
        # We only need current and previous j-layers
        dp_prev = [0] * (n + 2)  # dp[0][i] = 0
        dp_curr = [INF] * (n + 2)

        # Fill for j = 1..k
        for j in range(1, k + 1):
            # reset dp_curr
            for idx in range(n + 2):
                dp_curr[idx] = INF
            # compute from right to left
            for i in range(n - 1, -1, -1):
                # option1: skip starting window at i
                dp_curr[i] = dp_curr[i + 1]
                # option2: take window at i if valid
                if i <= n - x:
                    dp_curr[i] = min(dp_curr[i], costs[i] + dp_prev[i + x])
            # swap
            dp_prev, dp_curr = dp_curr, dp_prev

        res = dp_prev[0]
        return res if res < INF else -1