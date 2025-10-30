from typing import List
import heapq
from collections import defaultdict

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        m = n - x + 1                      # number of windows of length x
        
        # ---------- helper that maintains running median ---------
        class SlidingMedianCost:
            def __init__(self):
                self.low = []              # max-heap (as negative values)
                self.high = []             # min-heap
                self.delay = defaultdict(int)
                self.sum_low = 0           # sum of elements in low
                self.sum_high = 0          # sum of elements in high
                self.sz_low = 0
                self.sz_high = 0

            # remove elements that are marked for deletion
            def _prune(self, heap):
                while heap:
                    val = -heap[0] if heap is self.low else heap[0]
                    if self.delay.get(val, 0):
                        heapq.heappop(heap)
                        self.delay[val] -= 1
                        if self.delay[val] == 0:
                            del self.delay[val]
                    else:
                        break

            # balance the two heaps so that:
            #   sz_low >= sz_high and sz_low - sz_high <= 1
            def _balance(self):
                if self.sz_low > self.sz_high + 1:
                    v = -heapq.heappop(self.low)
                    self.sum_low -= v
                    self.sz_low -= 1
                    heapq.heappush(self.high, v)
                    self.sum_high += v
                    self.sz_high += 1
                    self._prune(self.low)
                elif self.sz_low < self.sz_high:
                    v = heapq.heappop(self.high)
                    self.sum_high -= v
                    self.sz_high -= 1
                    heapq.heappush(self.low, -v)
                    self.sum_low += v
                    self.sz_low += 1
                    self._prune(self.high)

            def add(self, num: int):
                if not self.low or num <= -self.low[0]:
                    heapq.heappush(self.low, -num)
                    self.sum_low += num
                    self.sz_low += 1
                else:
                    heapq.heappush(self.high, num)
                    self.sum_high += num
                    self.sz_high += 1
                self._balance()

            def remove(self, num: int):
                # mark for delayed deletion and adjust sums/sizes
                if num <= -self.low[0]:
                    self.delay[num] += 1
                    self.sum_low -= num
                    self.sz_low -= 1
                    if num == -self.low[0]:
                        self._prune(self.low)
                else:
                    self.delay[num] += 1
                    self.sum_high -= num
                    self.sz_high -= 1
                    if self.high and num == self.high[0]:
                        self._prune(self.high)
                self._balance()

            # current cost to make all elements equal to the median
            def cost(self) -> int:
                self._prune(self.low)
                self._prune(self.high)
                med = -self.low[0]
                return med * self.sz_low - self.sum_low + self.sum_high - med * self.sz_high
        # ----------------------------------------------------------------

        # compute cost for every window of length x
        smc = SlidingMedianCost()
        costs = [0] * m

        for i in range(x):
            smc.add(nums[i])
        costs[0] = smc.cost()

        for i in range(x, n):
            smc.add(nums[i])
            smc.remove(nums[i - x])
            costs[i - x + 1] = smc.cost()

        # ---------- DP: choose k non-overlapping windows --------------
        INF = 10 ** 20
        prev = [0] * m                 # dp for 0 subarrays (all zeros)

        for t in range(1, k + 1):
            cur = [INF] * m            # dp for t subarrays
            for i in range(m):
                # option1: not take window starting at i
                if i:
                    cur[i] = cur[i - 1]

                # option2: take window i
                if t == 1:
                    pre = 0
                else:
                    idx = i - x
                    pre = prev[idx] if idx >= 0 else INF
                if pre != INF:
                    cand = pre + costs[i]
                    if cand < cur[i]:
                        cur[i] = cand
            prev = cur

        return prev[-1]