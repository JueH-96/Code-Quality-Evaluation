from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # keeps track of whether each position is already marked
        marked = [False] * n
        
        # running sum of still-unmarked elements
        remaining_sum = sum(nums)
        
        # all elements sorted by value, then index  (so we can always take the global minimums)
        order = sorted((v, i) for i, v in enumerate(nums))
        ptr = 0                       # first position in `order` that has not been examined yet
        
        answers = []
        
        for idx, k in queries:
            # 1) mark the required index
            if not marked[idx]:
                marked[idx] = True
                remaining_sum -= nums[idx]
            
            # 2) mark the k smallest still-unmarked elements
            taken = 0
            while taken < k and ptr < n:
                val, pos = order[ptr]
                if not marked[pos]:
                    marked[pos] = True
                    remaining_sum -= val
                    taken += 1
                ptr += 1              # advance the pointer (each element inspected at most once)
            
            # 3) record answer
            answers.append(remaining_sum)
        
        return answers