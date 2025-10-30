class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Count how many queries cover each index
        counts = [0] * n
        for l, r in queries:
            for i in range(l, r + 1):
                counts[i] += 1
        
        # Check if it's possible to convert the array to a zero array
        for i in range(n):
            if counts[i] < nums[i]:
                return -1  # Not enough queries to reduce this index to 0
        
        # Initialize needed operations for each index
        needed = nums.copy()
        essential_queries = set()
        total_needed = sum(needed)
        
        # Greedily select queries until all elements can be reduced to 0
        while total_needed > 0:
            best_query = -1
            best_coverage = -1
            
            # Find the query that covers the most indices that still need operations
            for j in range(m):
                if j in essential_queries:
                    continue
                
                coverage = 0
                l, r = queries[j]
                for i in range(l, r + 1):
                    if needed[i] > 0:
                        coverage += 1
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_query = j
            
            # Mark the query as essential and update the needed operations
            essential_queries.add(best_query)
            l, r = queries[best_query]
            for i in range(l, r + 1):
                if needed[i] > 0:
                    needed[i] -= 1
                    total_needed -= 1
        
        # Return the number of queries we can remove
        return m - len(essential_queries)