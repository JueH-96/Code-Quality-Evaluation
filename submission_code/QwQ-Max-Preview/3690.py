class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        if not s:
            return 0
        
        # Compute the maximum run length in the original string
        max_run = 1
        current_run = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 1
        
        low, high = 1, max_run
        answer = max_run
        
        while low <= high:
            mid = (low + high) // 2
            if self.is_possible(s, mid, numOps):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
    
    def is_possible(self, s: str, L: int, numOps: int) -> bool:
        if L == 0:
            return False
        
        flips = 0
        current_char = s[0]
        current_length = 1
        
        for c in s[1:]:
            if c == current_char:
                current_length += 1
                if current_length > L:
                    flips += 1
                    # Flip the current character to the opposite
                    current_char = '1' if current_char == '0' else '0'
                    current_length = 1
                    if flips > numOps:
                        return False
            else:
                current_char = c
                current_length = 1
        
        return flips <= numOps