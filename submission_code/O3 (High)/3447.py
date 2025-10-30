class Solution:
    def clearDigits(self, s: str) -> str:
        # Use a list as a stack to keep the remaining characters
        stack = []
        
        for ch in s:
            if ch.isdigit():       # we have to remove this digit …
                if stack:          # … and the closest non-digit to its left
                    stack.pop()    # (guaranteed to exist by the problem)
            else:                  # it is a lowercase letter
                stack.append(ch)   # keep it for now
        
        # The stack now contains the characters that were never removed
        return ''.join(stack)