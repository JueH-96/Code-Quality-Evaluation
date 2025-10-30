from typing import List
import heapq
import collections
import math

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        m = n - x + 1  # number of segments
        
        # Two heaps for sliding window median.
        # small is a max-heap (store negatives) for the lower half.
        # large is a min-heap for the upper half.
        small = []
        large = []
        # Sums for each heap
        sum_small = 0
        sum_large = 0
        # Lazy deletion dictionaries
        delayed_small = collections.Counter()
        delayed_large = collections.Counter()

        # function to prune top of heap if it is marked for deletion.
        def prune(heap, delayed):
            # While the heap is non-empty and its top is marked for deletion.
            while heap and delayed[(heap[0] if heap is large else -heap[0])] > 0:
                if heap is large:
                    num = heapq.heappop(large)
                    delayed[num] -= 1
                else:
                    num = -heapq.heappop(small)
                    delayed[num] -= 1

        # rebalance the two heaps: ensure 
        # either len(small) == len(large) or len(small) == len(large) + 1
        def rebalance():
            nonlocal sum_small, sum_large
            if len(small) > len(large) + 1:
                # move top from small to large
                num = -heapq.heappop(small)
                sum_small -= num
                heapq.heappush(large, num)
                sum_large += num
                prune(small, delayed_small)
            elif len(small) < len(large):
                num = heapq.heappop(large)
                sum_large -= num
                heapq.heappush(small, -num)
                sum_small += num
                prune(large, delayed_large)
        
        # add a number to our data structure.
        def add(num):
            nonlocal sum_small, sum_large
            if not small or num <= -small[0]:
                heapq.heappush(small, -num)
                sum_small += num
            else:
                heapq.heappush(large, num)
                sum_large += num
            rebalance()
        
        # remove a number from our data structure.
        def remove(num):
            nonlocal sum_small, sum_large
            # decide which heap this num should be in.
            if small and num <= -small[0]:
                delayed_small[num] += 1
                sum_small -= num
                # If the top is this element, then prune.
                if num == -small[0]:
                    prune(small, delayed_small)
            else:
                delayed_large[num] += 1
                sum_large -= num
                if large and num == large[0]:
                    prune(large, delayed_large)
            rebalance()
        
        # get current median (which is top of small, because we ensure len(small) >= len(large))
        def getMedian():
            # small is not empty by design.
            return -small[0]
        
        # get current L1 cost in the window.
        def getCost():
            median = getMedian()
            # cost = median * count(small) - sum(small) + sum(large) - median * count(large)
            cnt_small = len(small)
            cnt_large = len(large)
            cost = median * cnt_small - sum_small + sum_large - median * cnt_large
            return cost

        # Precompute cost for every contiguous segment of size x.
        segCost = [0] * m
        # Build the first window.
        for i in range(x):
            add(nums[i])
        segCost[0] = getCost()
        for i in range(x, n):
            add(nums[i])
            remove(nums[i - x])
            # Clean up, though our rebalance and prune functions already do that.
            segCost[i - x + 1] = getCost()
        
        # Now use DP to choose k non-overlapping segments.
        # dp[i][j] = minimum cost using segments starting from index i to m-1 having already chosen j segments.
        # We use dp[i] for a particular j and then iterate for next j.
        INF = 10**18
        # dp array dimension: m+1 (for segments) x (k+1)
        dp = [[INF] * (k+1) for _ in range(m+1)]
        # base: if we require 0 segments, cost is 0.
        for i in range(m+1):
            dp[i][0] = 0

        # Process j = 1 to k, fill dp[i][j] from bottom up.
        # We iterate backward for i because segment i depends on dp[i+1] and dp[i+x].
        for j in range(1, k+1):
            # For i from m-1 down to 0.
            for i in range(m-1, -1, -1):
                # Option1: skip segment starting at i.
                best = dp[i+1][j]
                # Option2: take segment starting at i if possible.
                nxt = i + x
                if nxt <= m:
                    best = min(best, segCost[i] + dp[nxt][j-1])
                dp[i][j] = best

        return dp[0][k]