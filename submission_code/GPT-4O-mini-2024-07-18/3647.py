class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        total_decrement = sum(nums)
        
        # Create a list to store the effect of each query
        effect = [0] * (n + 1)
        
        # Calculate the effect of each query
        for l, r in queries:
            effect[l] += 1
            if r + 1 < n:
                effect[r + 1] -= 1
        
        # Calculate the prefix sum to get the number of queries affecting each index
        active_queries = [0] * n
        active_queries[0] = effect[0]
        for i in range(1, n):
            active_queries[i] = active_queries[i - 1] + effect[i]
        
        # Check if we can convert nums to a zero array using all queries
        total_possible_decrement = sum(min(nums[i], active_queries[i]) for i in range(n))
        
        if total_possible_decrement < total_decrement:
            return -1
        
        # Binary search to find the maximum number of removable queries
        left, right = 0, len(queries)
        while left < right:
            mid = (left + right) // 2
            # Create a temporary effect array for the first mid queries
            temp_effect = [0] * (n + 1)
            for i in range(mid):
                l, r = queries[i]
                temp_effect[l] += 1
                if r + 1 < n:
                    temp_effect[r + 1] -= 1
            
            # Calculate the active queries with the first mid queries removed
            temp_active_queries = [0] * n
            temp_active_queries[0] = temp_effect[0]
            for i in range(1, n):
                temp_active_queries[i] = temp_active_queries[i - 1] + temp_effect[i]
            
            # Check if we can still convert nums to a zero array
            total_possible_decrement = sum(min(nums[i], temp_active_queries[i]) for i in range(n))
            if total_possible_decrement >= total_decrement:
                left = mid + 1
            else:
                right = mid
        
        return left - 1