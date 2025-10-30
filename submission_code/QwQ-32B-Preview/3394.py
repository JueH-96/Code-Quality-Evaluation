class Solution:
    def minEnd(self, n: int, x: int) -> int:
        return (x + (n - 1)) | x