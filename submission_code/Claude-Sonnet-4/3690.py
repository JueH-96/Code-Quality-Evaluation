class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        def canAchieve(maxLen):
            operations = 0
            i = 0
            
            while i < n:
                # Find the length of current identical substring
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                
                currentLen = j - i
                
                # Calculate minimum flips needed to break this substring
                # so that no segment exceeds maxLen
                if currentLen > maxLen:
                    operations += currentLen // (maxLen + 1)
                
                i = j
            
            return operations <= numOps
        
        # Binary search on the answer
        left, right = 1, n
        
        while left < right:
            mid = (left + right) // 2
            
            if canAchieve(mid):
                right = mid
            else:
                left = mid + 1
        
        return left