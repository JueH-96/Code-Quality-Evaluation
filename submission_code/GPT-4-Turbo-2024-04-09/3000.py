class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        import heapq
        
        # We will use a min-heap to keep track of the smallest differences
        # and a sliding window approach to maintain the condition abs(i - j) >= x
        min_diff = float('inf')
        
        # Priority queue to store elements and their indices
        # Element format: (nums[i], i)
        min_heap = []
        
        # We start from index x and look back up to x indices to find the minimum difference
        for i in range(len(nums)):
            # Remove elements from the heap that are not at least x indices apart
            while min_heap and min_heap[0][1] <= i - x - 1:
                heapq.heappop(min_heap)
            
            # Check current value with the minimum value in the heap that is at least x indices apart
            if min_heap:
                # Calculate the absolute difference with the smallest element in the heap
                diff = abs(nums[i] - min_heap[0][0])
                min_diff = min(min_diff, diff)
            
            # Push current element into the heap
            heapq.heappush(min_heap, (nums[i], i))
        
        return min_diff