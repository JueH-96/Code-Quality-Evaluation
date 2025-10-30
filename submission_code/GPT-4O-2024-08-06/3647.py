from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Calculate the minimum number of decrements needed for each index
        min_decrements = [0] * n
        for num in nums:
            min_decrements[num] += 1
        
        # Calculate the effect of each query
        query_effect = [0] * (n + 1)
        for l, r in queries:
            query_effect[l] += 1
            if r + 1 < n:
                query_effect[r + 1] -= 1
        
        # Calculate the prefix sum to get the number of queries affecting each index
        active_queries = [0] * n
        active_queries[0] = query_effect[0]
        for i in range(1, n):
            active_queries[i] = active_queries[i - 1] + query_effect[i]
        
        # Check if it's possible to make the array zero with all queries
        for i in range(n):
            if min_decrements[i] > active_queries[i]:
                return -1
        
        # Binary search to find the maximum number of queries we can remove
        left, right = 0, m
        while left < right:
            mid = (left + right) // 2
            # Check if we can remove mid queries
            temp_effect = [0] * (n + 1)
            for i in range(mid, m):
                l, r = queries[i]
                temp_effect[l] += 1
                if r + 1 < n:
                    temp_effect[r + 1] -= 1
            
            temp_active = [0] * n
            temp_active[0] = temp_effect[0]
            for i in range(1, n):
                temp_active[i] = temp_active[i - 1] + temp_effect[i]
            
            possible = True
            for i in range(n):
                if min_decrements[i] > temp_active[i]:
                    possible = False
                    break
            
            if possible:
                left = mid + 1
            else:
                right = mid
        
        return left - 1