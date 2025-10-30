class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq

        n = len(nums)
        
        # Keep track of marked indices and the running sum of unmarked elements.
        marked = [False] * n
        total_sum = sum(nums)
        
        # A min-heap to efficiently find the smallest unmarked elements.
        # We store (value, index) tuples to handle tie-breaking by index.
        pq = [(nums[i], i) for i in range(n)]
        heapq.heapify(pq)
        
        answer = []
        
        # Process each query.
        for index, k in queries:
            # 1. Mark the element at the specified index if it's not already marked.
            if not marked[index]:
                marked[index] = True
                total_sum -= nums[index]
            
            # 2. Mark k smallest unmarked elements.
            elements_to_mark = k
            while elements_to_mark > 0 and pq:
                # Pop the smallest available element from the heap.
                value, i = heapq.heappop(pq)
                
                # If it's already marked (by index), skip it.
                if marked[i]:
                    continue
                
                # Otherwise, mark it, update the sum, and decrement the count.
                marked[i] = True
                total_sum -= value
                elements_to_mark -= 1
            
            # Append the current sum of unmarked elements to the answer.
            answer.append(total_sum)
            
        return answer