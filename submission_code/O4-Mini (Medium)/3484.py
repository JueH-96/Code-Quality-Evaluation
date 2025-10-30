class Solution:
    def getSmallestString(self, s: str) -> str:
        best = s
        n = len(s)
        for i in range(n - 1):
            # Check if s[i] and s[i+1] have the same parity
            if (int(s[i]) % 2) == (int(s[i+1]) % 2):
                arr = list(s)
                # Swap the adjacent same-parity digits
                arr[i], arr[i+1] = arr[i+1], arr[i]
                candidate = "".join(arr)
                # Keep the lexicographically smallest result
                if candidate < best:
                    best = candidate
        return best