class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        if n == 0:
            return s
        k %= n
        # For each position i, pick the character at (i + k) mod n
        return ''.join(s[(i + k) % n] for i in range(n))


# Example usages:
if __name__ == "__main__":
    sol = Solution()
    print(sol.getEncryptedString("dart", 3))  # Output: "tdar"
    print(sol.getEncryptedString("aaa", 1))   # Output: "aaa"