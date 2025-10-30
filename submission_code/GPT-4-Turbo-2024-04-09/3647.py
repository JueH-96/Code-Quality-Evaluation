class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        if not queries:
            return -1 if any(nums) else 0
        
        # Step 1: Calculate the total impact of all queries on each index
        impact = [0] * (n + 1)
        
        for l, r in queries:
            impact[l] += 1
            if r + 1 < n:
                impact[r + 1] -= 1
        
        # Accumulate the impact to get the actual number of times each index can be decremented
        for i in range(1, n):
            impact[i] += impact[i - 1]
        
        # Step 2: Check if we can turn nums to zero array with all queries
        for i in range(n):
            if nums[i] > impact[i]:
                return -1
        
        # Step 3: Use a greedy approach to determine the maximum number of queries that can be removed
        # We need to check the minimum number of times each index needs to be decremented
        min_needed = [0] * n
        for i in range(n):
            min_needed[i] = nums[i]
        
        # Calculate the maximum number of queries that can be removed
        # We need to find the minimum number of queries that are absolutely necessary
        # This is done by finding the tightest constraints
        max_removal = len(queries)
        query_usage = [0] * len(queries)
        
        for i, (l, r) in enumerate(queries):
            max_effect = 0
            for j in range(l, r + 1):
                max_effect = max(max_effect, min_needed[j])
            if max_effect > 0:
                query_usage[i] = max_effect
                for j in range(l, r + 1):
                    min_needed[j] -= max_effect
        
        # Check if we can still zero out the array
        if any(min_needed):
            return -1
        
        # Count the number of queries that are not used to their full potential
        for used in query_usage:
            if used == 0:
                max_removal -= 1
        
        return max_removal