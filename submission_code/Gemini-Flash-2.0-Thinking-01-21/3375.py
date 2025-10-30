import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        min_heap = []
        for i in range(len(coins)):
            heapq.heappush(min_heap, (coins[i], i))
        count = 0
        returned_values = set()
        last_returned = 0
        while count < k:
            current_val, coin_index = heapq.heappop(min_heap)
            if current_val not in returned_values:
                count += 1
                returned_values.add(current_val)
                last_returned = current_val
                if count == k:
                    return last_returned
            next_val = current_val + coins[coin_index]
            heapq.heappush(min_heap, (next_val, coin_index))
        return last_returned