from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        if n == 0:
            return m
        
        # Compute the difference array D
        D = []
        D.append(nums[0])
        for i in range(1, n):
            D.append(nums[i] - nums[i-1])
        
        sum_D = nums[-1]
        if sum_D != sum(D):
            return -1
        
        # Sort queries by r in descending order, then by l in ascending order
        sorted_queries = sorted(queries, key=lambda x: (-x[1], x[0]))
        
        # Initialize variables
        ptr = 0
        res = 0
        # Using a difference array to track the coverage
        delta = [0] * (n + 2)  # 1-based to n+1
        
        for i in range(n-1, -1, -1):
            # Calculate current coverage at position i
            current_coverage = 0
            for j in range(i+1):
                current_coverage += delta[j]
            required = D[i] - current_coverage
            if required < 0:
                return -1
            # Select required queries that cover i
            while required > 0 and ptr < m:
                l, r = sorted_queries[ptr]
                if r < i or l > i:
                    ptr += 1
                    continue
                # Apply this query
                delta[l] += 1
                if r + 1 <= n:
                    delta[r + 1] -= 1
                res += 1
                required -= 1
                ptr += 1
            if required > 0:
                return -1
        
        # Verify that the difference array matches D after applying all selected queries
        current = [0] * n
        current_delta = 0
        for i in range(n):
            current_delta += delta[i]
            current[i] = current_delta
        if current != D:
            return -1
        
        return m - res