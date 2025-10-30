class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        seen = set()
        res = 0
        count = 0
        i = 1
        while count < n:
            if i + (k - i) != k or k - i not in seen:
                res += i
                seen.add(i)
                count += 1
            i += 1
        return res