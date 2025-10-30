import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        total_sum = sum(nums)
        heap = []
        
        # Initialize the heap with (value, index) for all elements
        for idx, num in enumerate(nums):
            heapq.heappush(heap, (num, idx))
        
        result = []
        for idx, k in queries:
            if not marked[idx]:
                marked[idx] = True
                total_sum -= nums[idx]
            
            # Mark the k smallest unmarked elements
            count = 0
            while heap and count < k:
                num, i = heapq.heappop(heap)
                if not marked[i]:
                    marked[i] = True
                    total_sum -= num
                    count += 1
            result.append(total_sum)
        
        return result