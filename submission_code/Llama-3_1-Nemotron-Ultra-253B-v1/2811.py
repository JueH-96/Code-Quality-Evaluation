class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        included = []
        excluded = set()
        x = 1
        while len(included) < n:
            if x not in excluded:
                included.append(x)
                excluded.add(k - x)
            x += 1
        return sum(included)