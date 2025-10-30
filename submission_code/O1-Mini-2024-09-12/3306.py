from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        total = sum(nums)
        answer = []
        marked = [False] * n
        
        # Sort indices by (value, index)
        sorted_indices = sorted(range(n), key=lambda x: (nums[x], x))
        ptr = 0  # Pointer to the next smallest unmarked element
        
        for index, k in queries:
            # Mark the specified index if not already marked
            if not marked[index]:
                marked[index] = True
                total -= nums[index]
            
            # Mark k smallest unmarked elements
            count = 0
            while count < k and ptr < n:
                current_idx = sorted_indices[ptr]
                if not marked[current_idx]:
                    marked[current_idx] = True
                    total -= nums[current_idx]
                    count += 1
                ptr += 1
            
            answer.append(total)
        
        return answer