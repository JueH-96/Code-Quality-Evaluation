import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        total = sum(nums)
        heap = []
        
        # Initialize the heap with all elements
        for i in range(n):
            heapq.heappush(heap, (nums[i], i))
        
        result = []
        for idx, k in queries:
            if not marked[idx]:
                marked[idx] = True
                total -= nums[idx]
            
            # Mark the smallest k unmarked elements
            count = 0
            while heap and count < k:
                val, i = heapq.heappop(heap)
                if not marked[i]:
                    marked[i] = True
                    total -= val
                    count += 1
                else:
                    continue
            
            result.append(total)
        
        return result