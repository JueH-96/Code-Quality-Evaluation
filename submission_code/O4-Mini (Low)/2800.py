class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for ch in s:
            if stack:
                prev = stack[-1]
                # if the last two characters form "AB" or "CD", pop instead of push
                if (prev == 'A' and ch == 'B') or (prev == 'C' and ch == 'D'):
                    stack.pop()
                    continue
            stack.append(ch)
        return len(stack)


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minLength("ABFCACDB"))  # Expected output: 2
    print(sol.minLength("ACBBD"))     # Expected output: 5