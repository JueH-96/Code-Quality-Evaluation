class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        while any(c.isdigit() for c in s):
            # Find first digit
            digit_idx = -1
            for i in range(len(s)):
                if s[i].isdigit():
                    digit_idx = i
                    break
            
            # Find closest non-digit to left
            left_idx = -1
            for i in range(digit_idx-1, -1, -1):
                if not s[i].isdigit():
                    left_idx = i
                    break
                    
            # Delete digit and left character
            if left_idx != -1:
                s.pop(digit_idx)
                s.pop(left_idx)
            else:
                s.pop(digit_idx)
                
        return "".join(s)