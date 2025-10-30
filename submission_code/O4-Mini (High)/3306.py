from typing import List
import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # marked[i] == True means nums[i] has been marked
        marked = [False] * n

        # min‐heap of (value, index) for all elements
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)

        # track sum of unmarked elements
        total_unmarked = sum(nums)
        ans = []

        for idx, k in queries:
            # 1) mark the element at idx if not already marked
            if not marked[idx]:
                marked[idx] = True
                total_unmarked -= nums[idx]

            # 2) mark k smallest unmarked elements
            cnt = 0
            while cnt < k and heap:
                val, i = heapq.heappop(heap)
                if marked[i]:
                    # already marked (either by step 1 or earlier), skip
                    continue
                marked[i] = True
                total_unmarked -= val
                cnt += 1

            ans.append(total_unmarked)

        return ans