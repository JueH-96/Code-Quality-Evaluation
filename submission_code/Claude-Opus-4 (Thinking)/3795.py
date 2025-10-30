class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Check if nums is already all zeros
        if all(x == 0 for x in nums):
            return 0
        
        def canMakeZero(k):
            # Check if we can make nums zero using the first k queries
            
            # Precompute the maximum possible reduction for each index using queries[query_idx:k]
            max_reduction_from = [[0] * n for _ in range(k + 1)]
            for q in range(k - 1, -1, -1):
                l, r, val = queries[q]
                for i in range(n):
                    max_reduction_from[q][i] = max_reduction_from[q + 1][i]
                for i in range(l, r + 1):
                    max_reduction_from[q][i] += val
            
            # Check if it's even possible
            for i in range(n):
                if nums[i] > max_reduction_from[0][i]:
                    return False
            
            # Use recursion with memoization
            memo = {}
            
            def dfs(query_idx, current_nums):
                if all(x == 0 for x in current_nums):
                    return True
                if query_idx == k:
                    return False
                
                # Prune if it's impossible to reach zero from current state
                for i in range(n):
                    if current_nums[i] > max_reduction_from[query_idx][i]:
                        return False
                
                state = tuple(current_nums)
                if (query_idx, state) in memo:
                    return memo[(query_idx, state)]
                
                l, r, val = queries[query_idx]
                
                # Try all possible subsets of indices in [l, r] (including empty subset)
                for mask in range(1 << (r - l + 1)):
                    new_nums = current_nums[:]
                    valid = True
                    for i in range(r - l + 1):
                        if mask & (1 << i):
                            idx = l + i
                            new_nums[idx] -= val
                            if new_nums[idx] < 0:
                                valid = False
                                break
                    
                    if valid and dfs(query_idx + 1, new_nums):
                        memo[(query_idx, state)] = True
                        return True
                
                memo[(query_idx, state)] = False
                return False
            
            return dfs(0, nums[:])
        
        # Binary search for the minimum k
        left, right = 1, m
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canMakeZero(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result