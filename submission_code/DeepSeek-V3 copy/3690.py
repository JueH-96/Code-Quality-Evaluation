class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        if numOps >= n:
            return 1
        
        # Function to calculate the minimum possible length of the longest substring
        # after performing at most numOps flips
        def can_minimize(max_len):
            # We need to check if it's possible to make all runs of 0s and 1s <= max_len
            # by flipping at most numOps bits
            # We can treat the problem as finding the minimal number of flips to make all runs <= max_len
            # and then check if that number is <= numOps
            # To do this, we can iterate through the string and count the number of flips needed
            # to break long runs into smaller ones
            flips = 0
            current_run = 1
            for i in range(1, n):
                if s[i] == s[i-1]:
                    current_run += 1
                    if current_run > max_len:
                        flips += 1
                        current_run = 1
                else:
                    current_run = 1
            return flips <= numOps
        
        # Binary search for the minimal possible max_len
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