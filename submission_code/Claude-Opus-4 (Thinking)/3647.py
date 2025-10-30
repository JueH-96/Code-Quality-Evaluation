class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        import heapq
        
        n = len(nums)
        m = len(queries)
        
        # Sort queries by left endpoint
        sorted_queries = sorted((l, r, i) for i, (l, r) in enumerate(queries))
        
        used = 0
        query_idx = 0
        available = []  # max heap of (-right_end, original_index)
        active = []  # min heap of right_end of used queries
        
        for pos in range(n):
            # Remove expired active queries
            while active and active[0] < pos:
                heapq.heappop(active)
            
            # Add new available queries
            while query_idx < m and sorted_queries[query_idx][0] <= pos:
                l, r, orig_idx = sorted_queries[query_idx]
                heapq.heappush(available, (-r, orig_idx))
                query_idx += 1
            
            # Check if we need more queries
            current_coverage = len(active)
            need = nums[pos] - current_coverage
            
            if need > 0:
                # Select the best queries
                selected = 0
                while available and selected < need:
                    neg_r, orig_idx = heapq.heappop(available)
                    if -neg_r >= pos:  # This query covers the current position
                        heapq.heappush(active, -neg_r)
                        used += 1
                        selected += 1
                
                if selected < need:
                    return -1
        
        return m - used