class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Check if it's possible to convert nums to zero array
        def can_convert(selected_queries):
            remaining = nums.copy()
            for i in range(m):
                if not selected_queries[i]:
                    continue
                l, r = queries[i]
                for j in range(l, r + 1):
                    if remaining[j] > 0:
                        remaining[j] -= 1
            return all(x == 0 for x in remaining)
        
        # Try removing different combinations of queries
        def try_removal(count):
            def backtrack(pos, selected, remaining):
                if remaining == 0:
                    return can_convert(selected)
                if pos >= m:
                    return False
                
                # Skip current query (mark as removed)
                selected[pos] = False
                if backtrack(pos + 1, selected, remaining - 1):
                    return True
                
                # Use current query
                selected[pos] = True
                if backtrack(pos + 1, selected, remaining):
                    return True
                
                return False
            
            selected = [True] * m
            return backtrack(0, selected, count)
        
        # Binary search to find maximum number of removable queries
        left, right = 0, m
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if try_removal(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result