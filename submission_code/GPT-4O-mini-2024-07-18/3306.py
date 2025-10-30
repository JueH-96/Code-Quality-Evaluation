class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        
        n = len(nums)
        marked = [False] * n
        unmarked_sum = sum(nums)
        result = []
        
        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                unmarked_sum -= nums[index]
            
            # Create a min-heap of unmarked elements
            min_heap = []
            for i in range(n):
                if not marked[i]:
                    heapq.heappush(min_heap, (nums[i], i))
            
            # Mark k smallest unmarked elements
            for _ in range(k):
                if min_heap:
                    value, idx = heapq.heappop(min_heap)
                    if not marked[idx]:  # Check if it's still unmarked
                        marked[idx] = True
                        unmarked_sum -= value
            
            result.append(unmarked_sum)
        
        return result