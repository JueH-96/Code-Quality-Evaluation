class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        s = 0
        used = set()
        i = 1
        while len(used) < n:
            if target - i not in used:
                used.add(i)
                s += i
            i += 1
        return s