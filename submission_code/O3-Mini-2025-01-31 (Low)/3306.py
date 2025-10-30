from typing import List

class Fenw:
    def __init__(self, n):
        self.n = n
        self.ft = [0]*(n+1)
    
    def update(self, i: int, delta: int) -> None:
        # i is 0-indexed for our usage, but tree uses 1-indexing.
        i += 1
        while i <= self.n:
            self.ft[i] += delta
            i += i & -i

    def query(self, i: int) -> int:
        # returns sum from 0 to i (inclusive)
        s = 0
        i += 1
        while i > 0:
            s += self.ft[i]
            i -= i & -i
        return s

    def total(self) -> int:
        return self.query(self.n-1)
    
    def findKth(self, k: int) -> int:
        # Finds smallest index i such that prefix sum >= k.
        # Assumes k is 1-indexed (i.e. if k=1, find first element with value 1)
        idx = 0
        bit_mask = 1 << (self.n.bit_length())
        while bit_mask:
            t = idx + bit_mask
            if t <= self.n and self.ft[t] < k:
                k -= self.ft[t]
                idx = t
            bit_mask //= 2
        return idx  # already 0-indexed

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        
        # Create sorted_tuples: list of (value, original_index) sorted by (value, index)
        sorted_tuples = sorted((val, i) for i, val in enumerate(nums))
        
        # Build mapping: original index -> position in sorted_tuples
        orig_to_sorted = [0]*n
        for pos, (val, orig) in enumerate(sorted_tuples):
            orig_to_sorted[orig] = pos
        
        # Fenw tree to record unmarked elements (1 for unmarked, 0 for marked)
        fenw = Fenw(n)
        for i in range(n):
            fenw.update(i, 1)
            
        total_unmarked_sum = sum(nums)
        # Marking status for each original index
        marked = [False] * n
        
        answers = []
        
        # Process each query
        for query in queries:
            orig_index, k = query
            
            # Mark the queried index if not marked.
            if not marked[orig_index]:
                marked[orig_index] = True
                total_unmarked_sum -= nums[orig_index]
                pos_in_sorted = orig_to_sorted[orig_index]
                fenw.update(pos_in_sorted, -1)
            
            # Then mark k smallest unmarked elements by value, tie break by index.
            # We will mark them one by one.
            for _ in range(k):
                if fenw.total() == 0:
                    break
                # Find the first (smallest) unmarked element in sorted order (k=1).
                pos = fenw.findKth(1)
                # pos corresponds to sorted_tuples[pos]: (value, original index)
                val, orig = sorted_tuples[pos]
                # It is unmarked guaranteed by fenw tree.
                marked[orig] = True
                total_unmarked_sum -= nums[orig]
                fenw.update(pos, -1)
            # Append the current sum of unmarked elements.
            answers.append(total_unmarked_sum)
            
        return answers