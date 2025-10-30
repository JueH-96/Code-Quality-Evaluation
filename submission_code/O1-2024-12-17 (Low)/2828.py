class Solution:
    def smallestString(self, s: str) -> str:
        # Convert to a list for easier manipulation
        arr = list(s)
        n = len(arr)

        # Check if all characters are 'a'
        if all(ch == 'a' for ch in arr):
            # We must perform exactly one operation, so change the last character to 'z'
            arr[-1] = 'z'
            return "".join(arr)

        # Otherwise, find the first non-'a' character and start transforming
        i = 0
        # Skip leading 'a's
        while i < n and arr[i] == 'a':
            i += 1

        # Transform characters (subtract by 1) until we hit another 'a' or end of string
        while i < n and arr[i] != 'a':
            # 'a' -> 'z', 'b'->'a', 'c'->'b', ...
            arr[i] = chr((ord(arr[i]) - ord('a') - 1) % 26 + ord('a'))
            i += 1

        return "".join(arr)