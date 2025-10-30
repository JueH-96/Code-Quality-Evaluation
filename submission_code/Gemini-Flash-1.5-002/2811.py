class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        used = set()
        res = []
        cur = 1
        for i in range(n):
            while cur in used or (k - cur) in used:
                cur += 1
            res.append(cur)
            used.add(cur)
            used.add(k-cur) if k-cur >0 else None

        return sum(res)