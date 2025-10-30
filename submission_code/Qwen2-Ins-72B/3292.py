class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        indices = sorted((i, x) for i, x in enumerate(changeIndices))
        marked = [False] * n
        pq = []
        for i, x in indices:
            if not marked[i]:
                marked[i] = True
                heappush(pq, (nums[i], i))
            while pq and marked[pq[0][1]]:
                heappop(pq)
            if not pq:
                continue
            if pq[0][0] == 0:
                return i + 1
            pq[0] = (pq[0][0] - 1, pq[0][1])
        return -1 if any(not x for x in marked) else m