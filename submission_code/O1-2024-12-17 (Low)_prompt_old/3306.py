class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        
        n = len(nums)
        # Keep track of which elements are marked
        marked = [False] * n
        
        # Min-heap of (value, index) for all elements
        heap = [(val, i) for i, val in enumerate(nums)]
        heapq.heapify(heap)
        
        # Sum of all elements initially (none are marked yet)
        unmarked_sum = sum(nums)
        
        result = []
        
        for idx, k in queries:
            # Mark the element at index idx if not already marked
            if not marked[idx]:
                marked[idx] = True
                unmarked_sum -= nums[idx]
            
            # Pop from the min-heap up to k times, marking unmarked smallest (value, index)
            to_mark = k
            while to_mark > 0 and heap:
                val, i = heapq.heappop(heap)
                # If it's already marked, skip
                if marked[i]:
                    continue
                # Otherwise, mark it
                marked[i] = True
                unmarked_sum -= val
                to_mark -= 1
            
            # After the query, record the sum of unmarked elements
            result.append(unmarked_sum)
        
        return result