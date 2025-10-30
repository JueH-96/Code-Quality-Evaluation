class Solution:
    def smallestString(self, s: str) -> str:
        # Convert the string into a list of characters for easier manipulation
        chars = list(s)
        n = len(chars)
        
        # Step 1: Find the first index where the character is not 'a'
        start = -1
        for i in range(n):
            if chars[i] != 'a':
                start = i
                break
        
        # Step 2: If we did not find any character != 'a', then
        # all characters are 'a'. In that case, reduce the last character to 'z'
        # because we must perform exactly one operation.
        if start == -1:
            chars[n - 1] = 'z'
            return "".join(chars)
        
        # Step 3: From 'start', find how far to extend the substring
        # We stop when we see the next 'a' or reach the end of the string
        end = start
        while end < n and chars[end] != 'a':
            end += 1
        
        # Step 4: Reduce each character in the substring [start..end-1] by 1
        # ('b' -> 'a', 'c' -> 'b', ..., 'a' -> 'z')
        for j in range(start, end):
            # Shift one step backward in the alphabet
            # ord('b') -> 98, ord('a') -> 97, etc.
            # We map 'b' -> 'a', 'c' -> 'b', 'a' -> 'z', ...
            chars[j] = chr((ord(chars[j]) - ord('a') - 1) % 26 + ord('a'))
        
        # Step 5: Construct and return the resulting string
        return "".join(chars)