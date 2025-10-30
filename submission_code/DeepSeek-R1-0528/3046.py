class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        max_kept = 0
        if '0' in num:
            max_kept = max(max_kept, 1)
        patterns = [("0", "0"), ("2", "5"), ("5", "0"), ("7", "5")]
        for x, y in patterns:
            j = None
            for idx in range(n-1, -1, -1):
                if num[idx] == y:
                    j = idx
                    break
            if j is None:
                continue
            i = None
            for idx in range(j-1, -1, -1):
                if num[idx] == x:
                    i = idx
                    break
            if i is None:
                continue
            max_kept = max(max_kept, i + 2)
        return n - max_kept