from typing import List
import heapq
from collections import defaultdict

class DualHeap:
    def __init__(self, k):
        self.small = []  # max heap (store negatives)
        self.large = []  # min heap
        self.del_small = defaultdict(int)
        self.del_large = defaultdict(int)
        self.k = k
        self.size_small = 0
        self.size_large = 0
        self.sum_small = 0
        self.sum_large = 0

    def prune(self, heap):
        # Remove the top of 'heap' if it is marked for deletion
        if heap is self.small:
            while self.small and self.del_small[-self.small[0]] > 0:
                val = -heapq.heappop(self.small)
                self.del_small[val] -= 1
        else:
            while self.large and self.del_large[self.large[0]] > 0:
                val = heapq.heappop(self.large)
                self.del_large[val] -= 1

    def balance(self):
        # Maintain size_small >= size_large and size_small - size_large <= 1
        if self.size_small > self.size_large + 1:
            # move top of small to large
            val = -heapq.heappop(self.small)
            self.sum_small -= val
            self.size_small -= 1
            heapq.heappush(self.large, val)
            self.sum_large += val
            self.size_large += 1
            self.prune(self.small)
        elif self.size_small < self.size_large:
            # move top of large to small
            val = heapq.heappop(self.large)
            self.sum_large -= val
            self.size_large -= 1
            heapq.heappush(self.small, -val)
            self.sum_small += val
            self.size_small += 1
            self.prune(self.large)

    def get_median(self):
        # the median is top of small
        return -self.small[0]

    def insert(self, num):
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
            self.sum_small += num
            self.size_small += 1
        else:
            heapq.heappush(self.large, num)
            self.sum_large += num
            self.size_large += 1
        self.balance()

    def erase(self, num):
        # Lazy deletion: mark in the appropriate map
        median = self.get_median()
        if num <= median:
            self.del_small[num] += 1
            self.sum_small -= num
            self.size_small -= 1
            if -self.small[0] == num:
                self.prune(self.small)
        else:
            self.del_large[num] += 1
            self.sum_large -= num
            self.size_large -= 1
            if self.large and self.large[0] == num:
                self.prune(self.large)
        self.balance()

    def get_cost(self):
        # sum |ai - median|
        m = self.get_median()
        # for small part: sum(m - ai) = m*size_small - sum_small
        # for large part: sum(ai - m) = sum_large - m*size_large
        return m * self.size_small - self.sum_small + self.sum_large - m * self.size_large

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        m = n - x + 1  # number of possible segment starts
        # 1) Compute cost array
        dh = DualHeap(x)
        for i in range(x):
            dh.insert(nums[i])
        cost = [0] * m
        cost[0] = dh.get_cost()
        for i in range(x, n):
            dh.insert(nums[i])
            dh.erase(nums[i - x])
            cost[i - x + 1] = dh.get_cost()
        # 2) DP to pick k non-overlapping segments
        INF = 10**30
        # prev[j] = dp for j-1 segments, curr = for j segments
        prev = [0] * m  # dp[0][i] = 0 (zero segments cost 0)
        for seg in range(1, k + 1):
            curr = [INF] * m
            for i in range(m):
                if i > 0:
                    curr[i] = curr[i - 1]
                # choose segment starting at i
                prev_idx = i - x
                base = 0 if (seg == 1 and prev_idx < 0) else (prev[prev_idx] if prev_idx >= 0 else INF)
                if base < INF:
                    curr[i] = min(curr[i], cost[i] + base)
            prev = curr
        ans = prev[m - 1]
        return ans if ans < INF else -1