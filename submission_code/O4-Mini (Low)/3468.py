class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        # Normalize k to avoid unnecessary cycles
        k %= n
        # Build the encrypted string by taking the character k positions ahead (cyclically)
        res = []
        for i, ch in enumerate(s):
            res.append(s[(i + k) % n])
        return "".join(res)


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.getEncryptedString("dart", 3))  # Expected "tdar"
    print(sol.getEncryptedString("aaa", 1))   # Expected "aaa"