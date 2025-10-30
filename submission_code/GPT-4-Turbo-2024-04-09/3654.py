class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        import heapq
        
        # Priority queue to maximize the effect of each operation
        # We use negative values because heapq in Python is a min-heap by default
        heap = []
        for num in nums:
            heapq.heappush(heap, -num)
        
        # Apply Operation 2 (subtract k) up to op2 times
        for _ in range(op2):
            if not heap:
                break
            largest = -heapq.heappop(heap)
            if largest >= k:
                heapq.heappush(heap, -(largest - k))
            else:
                heapq.heappush(heap, -largest)
        
        # Apply Operation 1 (divide by 2, rounding up) up to op1 times
        for _ in range(op1):
            if not heap:
                break
            largest = -heapq.heappop(heap)
            heapq.heappush(heap, -(largest // 2 + largest % 2))
        
        # Calculate the sum of the modified array
        return -sum(heap)