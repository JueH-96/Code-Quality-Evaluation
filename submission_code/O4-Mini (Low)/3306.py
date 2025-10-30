from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Prepare list of elements sorted by (value, index)
        sorted_elems = sorted([(nums[i], i) for i in range(n)])
        # Marked flags
        marked = [False] * n
        # Pointer into sorted_elems for marking k smallest
        ptr = 0
        # Current total sum of unmarked elements
        total_sum = sum(nums)
        ans = []
        
        for idx, k in queries:
            # First, mark the specific index if unmarked
            if not marked[idx]:
                marked[idx] = True
                total_sum -= nums[idx]
            # Then mark k smallest unmarked
            while k > 0 and ptr < n:
                val, j = sorted_elems[ptr]
                ptr += 1
                if not marked[j]:
                    marked[j] = True
                    total_sum -= val
                    k -= 1
            # Record the sum after this query
            ans.append(total_sum)
        
        return ans