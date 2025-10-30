class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Create a priority queue to store the coins
        # The priority queue will be sorted in ascending order
        # The smallest element will be at the top of the queue
        import heapq
        queue = []
        for coin in coins:
            heapq.heappush(queue, coin)
        
        # Pop the smallest element k times
        for _ in range(k-1):
            heapq.heappop(queue)
        
        # The kth smallest element is the top of the queue
        return queue[0]