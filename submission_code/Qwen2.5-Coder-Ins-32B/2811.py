class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        used = set()
        i = 1
        while len(used) < n:
            if k - i not in used:
                used.add(i)
            i += 1
        return sum(used)