class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        
        n = len(nums)
        # Min-heap of (value, index) so we can always pick the smallest-value, smallest-index unmarked elements
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        sum_of_unmarked = sum(nums)
        marked = [False] * n  # Track which elements are marked
        
        answer = []
        
        for index_i, k_i in queries:
            # Force-mark index_i if not already marked
            if not marked[index_i]:
                marked[index_i] = True
                sum_of_unmarked -= nums[index_i]
            
            # Mark up to k_i smallest unmarked elements
            count_marked = 0
            while count_marked < k_i and heap:
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    sum_of_unmarked -= val
                    count_marked += 1
            
            # Record the sum of unmarked elements after this query
            answer.append(sum_of_unmarked)
        
        return answer