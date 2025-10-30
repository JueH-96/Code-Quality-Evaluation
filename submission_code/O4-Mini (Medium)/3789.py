from typing import List
import heapq
import bisect
from collections import defaultdict

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # We use INF = n+1 to denote "no conflict"
        INF = n + 1
        
        # Build a-map: for each a, list of b's in conflictingPairs
        a_map = defaultdict(list)
        for a, b in conflictingPairs:
            a_map[a].append(b)
        
        # M[l]  = the minimum b among pairs with a >= l
        # M2[l] = the second-minimum b among pairs with a >= l (or INF if none)
        M  = [INF] * (n + 2)
        M2 = [INF] * (n + 2)
        
        # We'll sweep l from n down to 1, inserting b's into a min-heap
        heap = []
        for l in range(n, 0, -1):
            # add all pairs that start at a = l
            for b in a_map.get(l, ()):
                heapq.heappush(heap, b)
            if not heap:
                M[l] = M2[l] = INF
            else:
                # extract the smallest
                b1 = heapq.heappop(heap)
                if not heap:
                    M[l] = b1
                    M2[l] = INF
                    heapq.heappush(heap, b1)
                else:
                    # extract the second smallest
                    b2 = heapq.heappop(heap)
                    M[l]  = b1
                    M2[l] = b2
                    # push both back
                    heapq.heappush(heap, b2)
                    heapq.heappush(heap, b1)
        
        # Compute base total of valid subarrays with all conflicts present:
        # for each start l, the farthest r we can go is (M[l] - 1),
        # so the count is max(0, (M[l]-1) - l + 1) = max(0, M[l] - l).
        base_total = 0
        for l in range(1, n + 1):
            base_total += (M[l] - l)
        
        # We need for each b-value to know the positions l where M[l] == b,
        # and the prefix-sum of M2[l] over those positions.
        pos = [[] for _ in range(n + 2)]
        psum = [[] for _ in range(n + 2)]
        
        for l in range(1, n + 1):
            b = M[l]
            pos[b].append(l)
            psum[b].append(M2[l])
        
        # Build prefix-sums of M2 for each b
        for b in range(n + 2):
            arr = psum[b]
            for i in range(1, len(arr)):
                arr[i] += arr[i-1]
        
        # Now try removing each conflicting pair (a,b):
        # it only affects l ≤ a where M[l] == b.
        # For those l, M[l] would rise to M2[l], so each such l gains
        # (M2[l] - b) extra valid extensions.
        ans = base_total
        for a, b in conflictingPairs:
            lst = pos[b]
            if not lst:
                continue
            # count how many l in lst are ≤ a
            idx = bisect.bisect_right(lst, a)
            if idx == 0:
                continue
            count = idx
            sum_m2 = psum[b][idx - 1]
            delta = sum_m2 - count * b
            ans = max(ans, base_total + delta)
        
        return ans