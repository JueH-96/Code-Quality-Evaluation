class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return str(k * (9 // k))
        half = '9' * (n // 2)
        for i in range(int(half), -1, -1):
            cand = int(str(i) + str(i)[::-1])
            if cand % k == 0:
                return str(cand)
        return ''