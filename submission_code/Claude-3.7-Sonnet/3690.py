class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        def can_achieve(max_len):
            # Check if we can ensure no run is of length > max_len using at most numOps operations
            s_copy = list(s)
            ops = 0
            
            for i in range(n):
                # Check the run length up to the current position
                run_length = 1
                j = i - 1
                while j >= 0 and s_copy[j] == s_copy[i]:
                    run_length += 1
                    j -= 1
                
                # If the run is too long, flip the current character
                if run_length > max_len:
                    s_copy[i] = '0' if s_copy[i] == '1' else '1'
                    ops += 1
                    if ops > numOps:
                        return False
            
            return True
        
        # Binary search for the minimum possible value of the maximum run length
        left, right = 1, n
        result = n
        
        while left <= right:
            mid = (left + right) // 2
            if can_achieve(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result