class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        s = set()
        res = 0
        cur = 1
        while len(s) < n:
            if k - cur not in s:
                s.add(cur)
                res += cur
            cur += 1
        return res