class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        import heapq
        
        n = len(nums)
        queries.sort()  # Sort queries by start position
        
        # Max heap for queries that can be used (store negative end positions)
        available = []
        # Current active queries (those being used)
        active = []
        
        query_idx = 0
        current_coverage = 0
        
        for i in range(n):
            # Add all queries that start at position i to available queries
            while query_idx < len(queries) and queries[query_idx][0] == i:
                heapq.heappush(available, -queries[query_idx][1])  # negative for max heap
                query_idx += 1
            
            # Remove expired active queries
            while active and active[0] < i:
                heapq.heappop(active)
            
            current_coverage = len(active)
            
            # If we need more coverage for position i
            while current_coverage < nums[i]:
                # Try to use an available query
                if not available:
                    return -1  # Cannot cover this position
                
                end_pos = -heapq.heappop(available)
                if end_pos < i:
                    continue  # This query cannot cover position i
                
                # Use this query
                heapq.heappush(active, end_pos)
                current_coverage += 1
        
        # Return the number of unused queries
        return len(available)