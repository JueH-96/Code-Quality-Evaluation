class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Sort queries by end position, then by start position
        queries_with_idx = [(l, r, i) for i, (l, r) in enumerate(queries)]
        queries_with_idx.sort(key=lambda x: (x[1], x[0]))
        
        remaining = nums[:]
        used = 0
        
        for l, r, orig_idx in queries_with_idx:
            # Check if we need this query
            needed = False
            for i in range(l, r + 1):
                if remaining[i] > 0:
                    needed = True
                    break
            
            if needed:
                # Use this query
                used += 1
                for i in range(l, r + 1):
                    if remaining[i] > 0:
                        remaining[i] -= 1
        
        # Check if all positions are zero
        if all(x == 0 for x in remaining):
            return m - used
        else:
            return -1