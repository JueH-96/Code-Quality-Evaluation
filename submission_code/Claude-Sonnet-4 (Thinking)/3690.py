class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def canAchieve(target):
            # Find all runs of consecutive identical characters
            runs = []
            i = 0
            n = len(s)
            while i < n:
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                runs.append(j - i)
                i = j
            
            # Calculate total flips needed
            total_flips = 0
            for run_length in runs:
                if run_length > target:
                    flips_needed = run_length // (target + 1)
                    total_flips += flips_needed
            
            return total_flips <= numOps
        
        # Binary search on the target
        left, right = 1, len(s)
        while left < right:
            mid = (left + right) // 2
            if canAchieve(mid):
                right = mid
            else:
                left = mid + 1
        return left