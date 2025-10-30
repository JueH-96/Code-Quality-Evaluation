import heapq
from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)

        def can_mark_all(k: int) -> bool:
            # k is the number of seconds available (from 1 to k)
            
            # Step 1: Find the latest occurrence for each index within `k` seconds.
            # last_mark_sec[j] stores the latest second s (1-indexed) <= k where changeIndices[s-1] == j
            last_mark_sec = [0] * (n + 1) # 1-indexed for indices 1 to n
            for s in range(1, k + 1):
                idx = changeIndices[s - 1]
                last_mark_sec[idx] = s
            
            # Step 2: Check if all indices (1 to n) can be marked.
            # If any index j does not appear in changeIndices up to second k, it cannot be marked.
            for j in range(1, n + 1):
                if last_mark_sec[j] == 0:
                    return False

            # Step 3: Simulate the process in reverse (from k down to 1).
            # Use a max-priority queue (implemented with min-heap for negative values)
            # to store the values of nums[idx] that still need to be decremented.
            # We add an index to the PQ when we encounter its last mark time.
            # This means its decrements must be completed by this time.
            
            pq = [] # Max-heap: stores (-value, original_0_indexed_index)
            
            # Iterate seconds from k down to 1
            for s in range(k, 0, -1):
                idx_1_indexed = changeIndices[s - 1] # The 1-indexed index mentioned in changeIndices

                if s == last_mark_sec[idx_1_indexed]:
                    # This second 's' MUST be used to mark idx_1_indexed.
                    # We add this index to our "active set" (via PQ) because its value needs to be zeroed.
                    # Its current value is nums[idx_1_indexed - 1].
                    heapq.heappush(pq, (-nums[idx_1_indexed - 1], idx_1_indexed - 1))
                else:
                    # This second 's' is available for a decrement operation.
                    # We use it to decrement the value of an active index that currently
                    # requires the most decrements.
                    if pq:
                        current_val_neg, original_0_indexed_idx = heapq.heappop(pq)
                        current_val = -current_val_neg
                        
                        if current_val > 0:
                            # Decrement this value.
                            current_val -= 1
                            heapq.heappush(pq, (-current_val, original_0_indexed_idx))
                        else:
                            # Value is already 0, push it back as 0.
                            heapq.heappush(pq, (0, original_0_indexed_idx))
            
            # Step 4: After processing all k seconds, check if all values in the PQ are 0.
            # If any value is still > 0, it means we could not zero it out in time.
            for val_neg, _ in pq:
                if -val_neg > 0:
                    return False
            
            return True

        # Binary search for the earliest second `k`
        low = 1
        high = m
        ans = -1

        while low <= high:
            mid = low + (high - low) // 2
            if can_mark_all(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans