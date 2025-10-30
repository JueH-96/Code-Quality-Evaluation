class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        
        n = len(nums)
        is_marked = [False] * n
        unmarked_sum = sum(nums)
        answer = []
        
        # Create a min-heap of (value, index) pairs for all elements
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        for index_i, k_i in queries:
            # Mark the element at index_i if it's not already marked
            if not is_marked[index_i]:
                is_marked[index_i] = True
                unmarked_sum -= nums[index_i]
            
            # Mark the k_i smallest unmarked elements
            marked_count = 0
            while heap and marked_count < k_i:
                val, idx = heapq.heappop(heap)
                if not is_marked[idx]:
                    is_marked[idx] = True
                    unmarked_sum -= val
                    marked_count += 1
            
            answer.append(unmarked_sum)
        
        return answer