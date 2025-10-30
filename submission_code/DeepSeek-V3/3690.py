class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        if numOps >= n:
            return 1
        
        # Function to calculate the minimum possible length
        def can_minimize(target):
            # We need to find if we can make all runs of 0s and 1s <= target
            # by flipping at most numOps characters
            # We can treat the problem as finding the minimal number of flips to make all runs <= target
            # and then check if that number is <= numOps
            # To do this, we can iterate through the string and count the number of flips needed
            # to make each run <= target
            flips = 0
            current_run = 1
            for i in range(1, n):
                if s[i] == s[i-1]:
                    current_run += 1
                    if current_run > target:
                        flips += 1
                        current_run = 1
                else:
                    current_run = 1
            return flips <= numOps
        
        # Binary search for the minimal possible length
        left = 1
        right = n
        result = n
        while left <= right:
            mid = (left + right) // 2
            if can_minimize(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        return result