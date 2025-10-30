class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        def canMakeZero(k):
            # Try to make array zero using first k queries
            n = len(nums)
            remaining = nums[:]
            
            # For each position, we need to reduce it to 0
            for i in range(n):
                if remaining[i] == 0:
                    continue
                    
                # Find queries that can affect position i
                needed = remaining[i]
                
                # Go through queries that can affect position i
                for j in range(k):
                    l, r, val = queries[j]
                    if l <= i <= r and needed > 0:
                        # Use this query to reduce position i
                        reduction = min(needed, val)
                        needed -= reduction
                        
                        # Update the query's remaining capacity
                        queries[j][2] -= reduction
                
                if needed > 0:
                    # Restore query values for next attempt
                    for j in range(k):
                        queries[j][2] = queries[j][2] if len(queries[j]) > 3 else queries[j][2]
                    return False
            
            # Restore query values for next attempt
            for j in range(k):
                l, r, val = queries[j]
                queries[j] = [l, r, val]
            
            return True
        
        # Better approach: use difference array and greedy
        def canMakeZeroOptimal(k):
            n = len(nums)
            target = nums[:]
            
            # Use greedy approach: process from left to right
            for i in range(n):
                if target[i] <= 0:
                    continue
                
                needed = target[i]
                
                # Find queries that can help at position i
                for j in range(k):
                    l, r, val = queries[j]
                    if l <= i <= r and needed > 0:
                        # Use minimum of what we need and what query can provide
                        use = min(needed, val)
                        needed -= use
                        
                        # Apply this reduction to all positions in range [i, r]
                        for pos in range(i, r + 1):
                            target[pos] -= use
                        
                        # Update query capacity
                        queries[j][2] -= use
                
                if target[i] > 0:
                    # Restore queries
                    for j in range(k):
                        queries[j] = [queries[j][0], queries[j][1], queries[j][2] + (queries[j][2] if len(str(queries[j])) > 10 else 0)]
                    return False
            
            # Restore queries
            original_queries = [[l, r, val] for l, r, val in queries]
            for j in range(k):
                queries[j] = original_queries[j]
            
            return True
        
        # Simpler greedy approach
        def canMakeZeroSimple(k):
            target = nums[:]
            available_queries = [list(q) for q in queries[:k]]
            
            for i in range(len(nums)):
                while target[i] > 0:
                    # Find a query that can reduce position i
                    found = False
                    for j, (l, r, val) in enumerate(available_queries):
                        if l <= i <= r and val > 0:
                            reduction = min(target[i], val)
                            target[i] -= reduction
                            available_queries[j][2] -= reduction
                            found = True
                            break
                    
                    if not found:
                        return False
            
            return True
        
        # Binary search on answer
        left, right = 0, len(queries)
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canMakeZeroSimple(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result