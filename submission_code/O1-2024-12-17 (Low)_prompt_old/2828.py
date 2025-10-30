class Solution:
    def smallestString(self, s: str) -> str:
        # Convert to list of characters for easier manipulation
        arr = list(s)
        n = len(arr)

        # Find the first character from the left that is not 'a'
        idx = 0
        while idx < n and arr[idx] == 'a':
            idx += 1

        # If all characters are 'a', we must shift exactly one substring
        # => pick the last character and shift it to 'z'
        if idx == n:
            arr[-1] = 'z'
            return "".join(arr)

        # From the first non-'a' character, shift until we hit 'a' or end
        while idx < n and arr[idx] != 'a':
            # Shift current character by 1 backward in the alphabet:
            # e.g., 'b' -> 'a', 'a' -> 'z'
            if arr[idx] == 'a':
                arr[idx] = 'z'
            else:
                arr[idx] = chr(ord(arr[idx]) - 1)
            idx += 1

        return "".join(arr)