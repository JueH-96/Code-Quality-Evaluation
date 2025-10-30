class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def canAchieve(target):
            flips_needed = 0
            i = 0
            n = len(s)
            
            while i < n:
                j = i
                # Find the end of current sequence of identical characters
                while j < n and s[j] == s[i]:
                    j += 1
                
                length = j - i
                if length > target:
                    # Calculate minimum flips needed to ensure no segment > target
                    flips_needed += length // (target + 1)
                
                i = j
            
            return flips_needed <= numOps
        
        # Binary search on the answer
        left, right = 1, len(s)
        result = len(s)
        
        while left <= right:
            mid = (left + right) // 2
            if canAchieve(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result