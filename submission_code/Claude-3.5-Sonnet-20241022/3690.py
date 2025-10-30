class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        def can_achieve_length(target_len: int) -> bool:
            # For each window of size target_len, check if we can make all chars same
            # using at most numOps operations
            for i in range(n - target_len + 1):
                window = s[i:i + target_len]
                # Try making all chars 0
                ops_for_zero = sum(1 for c in window if c == '1')
                # Try making all chars 1
                ops_for_one = sum(1 for c in window if c == '0')
                if min(ops_for_zero, ops_for_one) <= numOps:
                    return True
            return False
        
        # If numOps is 0, return the length of longest existing identical substring
        if numOps == 0:
            curr_len = 1
            max_len = 1
            for i in range(1, n):
                if s[i] == s[i-1]:
                    curr_len += 1
                else:
                    curr_len = 1
                max_len = max(max_len, curr_len)
            return max_len
        
        # Binary search on the answer
        left, right = 1, n + 1
        while left < right:
            mid = (left + right) // 2
            if can_achieve_length(mid):
                left = mid + 1
            else:
                right = mid
                
        # Return left - 1 as we want the minimum length
        return left - 1