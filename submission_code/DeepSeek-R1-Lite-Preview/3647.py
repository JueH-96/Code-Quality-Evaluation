from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        q = len(queries)
        n = len(nums)
        
        # Sort queries by ending index in descending order
        queries.sort(key=lambda x: -x[1])
        
        diff = [0] * (n + 2)
        pos = 0
        queries_used = 0
        i = 0
        
        while pos < n:
            # Compute current coverage at pos using prefix sum up to pos
            coverage = 0
            for j in range(pos):
                coverage += diff[j]
            coverage += diff[pos]
            
            if coverage < nums[pos]:
                # Need to cover pos
                # Find the first query that covers pos
                while i < q and queries[i][0] > pos:
                    i += 1
                if i < q and queries[i][0] <= pos:
                    # Select this query
                    queries_used += 1
                    # Increment diff[l] and decrement diff[r+1]
                    l = queries[i][0]
                    r = queries[i][1]
                    diff[l] += 1
                    if r + 1 < n:
                        diff[r + 1] -= 1
                    # Move pos to r + 1
                    pos = r + 1
                else:
                    # No query can cover pos
                    return -1
            else:
                pos += 1
        
        # After selecting queries, compute the prefix sum to get coverage
        prefix = 0
        for i in range(n):
            prefix += diff[i]
            if prefix < nums[i]:
                return -1
        
        return q - queries_used