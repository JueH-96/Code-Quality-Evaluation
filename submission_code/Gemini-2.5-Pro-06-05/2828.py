class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        chars = list(s)

        # Find the start of the first segment of non-'a' characters.
        # We must skip any leading 'a's because changing them to 'z'
        # would make the string lexicographically larger.
        i = 0
        while i < n and chars[i] == 'a':
            i += 1

        # If the entire string consists of 'a's, we must perform an operation.
        # To get the lexicographically smallest string, we change the last
        # character from 'a' to 'z'. e.g., "aaa" -> "aaz"
        if i == n:
            chars[n - 1] = 'z'
            return "".join(chars)

        # The operation must start at index `i` to achieve the lexicographically
        # smallest result. We continue the operation as long as we encounter
        # non-'a' characters. We stop at the first 'a' because changing it to 'z'
        # would make the string larger.
        j = i
        while j < n and chars[j] != 'a':
            # Decrement the character.
            chars[j] = chr(ord(chars[j]) - 1)
            j += 1
        
        return "".join(chars)