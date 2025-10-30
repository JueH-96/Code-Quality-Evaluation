class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        s = set()
        res = 0
        i = 1
        while len(s) < n:
            if k - i not in s:
                s.add(i)
                res += i
            i += 1
        return res