class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        
        # Sort queries by start position
        queries_with_idx = [(l, r, i) for i, (l, r) in enumerate(queries)]
        queries_with_idx.sort()
        
        # Track which queries we must use
        used = set()
        
        # Process each position
        query_idx = 0
        import heapq
        available = []  # max heap of end positions (negative for max heap)
        
        for pos in range(n):
            # Add all queries that start at or before current position
            while query_idx < len(queries_with_idx) and queries_with_idx[query_idx][0] <= pos:
                l, r, idx = queries_with_idx[query_idx]
                if r >= pos:  # Only add if it covers current position
                    heapq.heappush(available, (-r, idx))
                query_idx += 1
            
            # Remove queries that no longer cover current position
            while available and -available[0][0] < pos:
                heapq.heappop(available)
            
            # Use queries to cover current position
            needed = nums[pos]
            temp_available = []
            
            while needed > 0 and available:
                neg_r, idx = heapq.heappop(available)
                r = -neg_r
                if r >= pos:  # Can use this query
                    used.add(idx)
                    needed -= 1
                    temp_available.append((neg_r, idx))
            
            # Put back unused queries
            for item in temp_available:
                heapq.heappush(available, item)
            
            if needed > 0:
                return -1
        
        return len(queries) - len(used)