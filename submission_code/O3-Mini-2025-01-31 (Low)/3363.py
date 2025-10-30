from typing import List
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # We'll use a max heap (using negative counts) to track the maximum count efficiently.
        # Dictionary to record the current count for each id.
        id_count = {}
        max_heap = []  # will store tuples (-count, id)
        result = []
        
        for id_val, change in zip(nums, freq):
            old_count = id_count.get(id_val, 0)
            new_count = old_count + change
            id_count[id_val] = new_count
            
            # Push the updated count for this id into the heap.
            heapq.heappush(max_heap, (-new_count, id_val))
            
            # Clean the heap:
            # Because we push updates repeatedly for the same id, the heap may contain stale values.
            # Remove them until the top of the heap is consistent with our id_count.
            while max_heap:
                current_count, current_id = max_heap[0]
                # current_count is negative; its positive value should match id_count[current_id]
                if -current_count == id_count[current_id]:
                    break
                heapq.heappop(max_heap)
            
            # The maximum frequent count (if the collection is empty, max_heap might be empty)
            best = -max_heap[0][0] if max_heap else 0
            result.append(best)
        
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    nums1 = [2,3,2,1]
    freq1 = [3,2,-3,1]
    print(sol.mostFrequentIDs(nums1, freq1))  # Expected output: [3,3,2,2]

    # Example 2
    nums2 = [5,5,3]
    freq2 = [2,-2,1]
    print(sol.mostFrequentIDs(nums2, freq2))  # Expected output: [2,0,1]