class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        # Find all groups of consecutive identical characters
        groups = []
        i = 0
        n = len(s)
        
        while i < n:
            j = i
            while j < n and s[j] == s[i]:
                j += 1
            groups.append(j - i)
            i = j
        
        # Binary search on the answer
        left, right = 1, max(groups)
        
        def canAchieve(maxLen):
            # Check if we can make all groups have length <= maxLen
            # using at most numOps operations
            ops = 0
            for length in groups:
                if length > maxLen:
                    # Number of flips needed to break this group into pieces of size <= maxLen
                    # We need to place flips at intervals of maxLen
                    ops += (length - 1) // maxLen
            return ops <= numOps
        
        while left < right:
            mid = (left + right) // 2
            if canAchieve(mid):
                right = mid
            else:
                left = mid + 1
        
        return left