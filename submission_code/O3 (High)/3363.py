from typing import List
import heapq


class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        cnt = {}                # current counter of every id
        heap = []               # max-heap with pairs (-counter, id)
        ans = []

        for val, delta in zip(nums, freq):
            # update counter
            cnt[val] = cnt.get(val, 0) + delta

            if cnt[val] > 0:
                # store the new (counter, id) in the heap
                heapq.heappush(heap, (-cnt[val], val))
            else:                   # counter back to zero – remove id
                del cnt[val]

            # obtain current maximum
            while heap:
                neg_c, id_ = heap[0]
                # valid if id_ still exists and counter matches
                if id_ in cnt and cnt[id_] == -neg_c:
                    ans.append(cnt[id_])
                    break
                # otherwise it is stale -> discard
                heapq.heappop(heap)

            if not heap:            # collection empty
                ans.append(0)

        return ans