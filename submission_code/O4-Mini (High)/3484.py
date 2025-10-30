class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert string to list for easy swapping
        arr = list(s)
        n = len(arr)
        # Try each adjacent pair exactly once
        for i in range(n - 1):
            a = ord(arr[i]) - ord('0')
            b = ord(arr[i+1]) - ord('0')
            # Check if same parity and swapping makes it smaller
            if (a % 2) == (b % 2) and a > b:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                return "".join(arr)
        # No beneficial swap found
        return s