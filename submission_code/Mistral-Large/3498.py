from typing import List
from collections import defaultdict

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        freq = defaultdict(int)
        cost = [0] * (k + 1)

        for i in range(k):
            freq.clear()
            for j in range(i, n, k):
                freq[nums[j]] += 1
                freq[nums[n - 1 - j]] += 1

            min_cost = float('inf')
            for x in range(k + 1):
                cost[x] += 2 - freq[x]
                min_cost = min(min_cost, cost[x])

            if i < k - 1:
                for x in range(k + 1):
                    cost[x] = min_cost

        return min(cost)