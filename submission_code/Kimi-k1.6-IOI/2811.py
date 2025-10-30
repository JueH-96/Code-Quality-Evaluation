class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        selected = set()
        x = 1
        while len(selected) < n:
            if (k - x) not in selected:
                selected.add(x)
            x += 1
        return sum(selected)