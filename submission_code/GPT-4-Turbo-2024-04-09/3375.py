class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        import heapq
        
        # Min-heap to store the next smallest multiples
        heap = []
        
        # Initialize the heap with the first multiple of each coin
        for coin in coins:
            heapq.heappush(heap, coin)
        
        # We need to find the k-th smallest, so we pop from the heap k times
        smallest = 0
        for _ in range(k):
            smallest = heapq.heappop(heap)
            # For each coin, if the current smallest is its multiple, push the next multiple
            for coin in coins:
                next_multiple = smallest + coin
                # Push the next multiple to the heap
                heapq.heappush(heap, next_multiple)
                # Since all coins are distinct and we only push multiples of the current smallest,
                # we can break after pushing to avoid duplicates in the heap
                if smallest % coin == 0:
                    break
        
        return smallest