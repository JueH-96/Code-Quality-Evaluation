class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        total_sum = sum(nums)
        result = []
        
        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                total_sum -= nums[index]
            
            while k > 0 and heap:
                value, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    total_sum -= value
                    k -= 1
            
            result.append(total_sum)
        
        return result