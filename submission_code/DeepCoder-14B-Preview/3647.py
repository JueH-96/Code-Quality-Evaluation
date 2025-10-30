class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        req = nums.copy()
        # Sort queries in decreasing order of their right end
        queries.sort(key=lambda x: (-x[1], -x[0]))
        count = 0  # Number of queries used (kept)
        
        for l, r in queries:
            # Check if any element in [l, r] is still positive
            has_positive = False
            for i in range(l, r + 1):
                if req[i] > 0:
                    req[i] -= 1
                    has_positive = True
            if has_positive:
                count += 1
        
        # Check if all elements are non-positive
        for i in range(len(req)):
            if req[i] > 0:
                return -1
        
        # The maximum number of queries that can be removed is total queries minus the ones used
        return len(queries) - count