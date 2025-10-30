class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        n = len(nums)
        m = len(queries)
        sum_unmarked = sum(nums)
        marked = [False] * n
        heap = []
        for i in range(n):
            heapq.heappush(heap, (nums[i], i))
        answer = []
        for index_i, k_i in queries:
            if not marked[index_i]:
                marked[index_i] = True
                sum_unmarked -= nums[index_i]
            for _ in range(k_i):
                while heap and marked[heap[0][1]]:
                    heapq.heappop(heap)
                if not heap:
                    break
                val, idx = heapq.heappop(heap)
                marked[idx] = True
                sum_unmarked -= val
            answer.append(sum_unmarked)
        return answer