class Solution:
    def earliestSecondToMarkIndices(self, nums, changeIndices):
        n = len(nums)
        m = len(changeIndices)
        nums = [0] + nums
        changeIndices = [0] + changeIndices
        marked = [0] * (n + 1)
        heap = []
        for i in range(1, n + 1):
            heap.append((-nums[i], i))
        import heapq
        heapq.heapify(heap)
        for i in range(1, m + 1):
            while heap and marked[heap[0][1]]:
                heapq.heappop(heap)
            if not heap or -heap[0][0] > i:
                return -1
            if nums[changeIndices[i]] == 0 and not marked[changeIndices[i]]:
                marked[changeIndices[i]] = 1
            else:
                if heap:
                    val, idx = heapq.heappop(heap)
                    if -val > i:
                        heapq.heappush(heap, (val, idx))
                    else:
                        heapq.heappush(heap, (val + 1, idx))
        return m if sum(marked) == n else -1