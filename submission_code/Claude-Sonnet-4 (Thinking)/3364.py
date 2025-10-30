class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(pos, seg):
            # pos: starting position for current segment
            # seg: current segment index in andValues
            
            if seg == m:
                return 0 if pos == n else float('inf')
            
            if pos >= n or m - seg > n - pos:
                return float('inf')
            
            result = float('inf')
            current_and = nums[pos]
            
            # Try all possible ending positions for this segment
            for end in range(pos, n):
                if end > pos:
                    current_and &= nums[end]
                
                if current_and == andValues[seg]:
                    result = min(result, nums[end] + dp(end + 1, seg + 1))
                
                # Early termination: if current_and cannot become andValues[seg]
                # because it's missing required 1-bits
                if (current_and & andValues[seg]) != andValues[seg]:
                    break
            
            return result
        
        result = dp(0, 0)
        return result if result != float('inf') else -1