class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        import heapq

        # Min-heap to store the next possible smallest amounts
        min_heap = []
        
        # Initialize the heap with the first multiple of each coin
        for coin in coins:
            heapq.heappush(min_heap, (coin, coin))
        
        # Variable to store the kth smallest amount
        kth_smallest = 0
        
        # Extract the smallest element from the heap k times
        for _ in range(k):
            kth_smallest, coin = heapq.heappop(min_heap)
            # Push the next multiple of the current coin
            heapq.heappush(min_heap, (kth_smallest + coin, coin))
        
        return kth_smallest