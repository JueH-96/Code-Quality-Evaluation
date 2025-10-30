class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result = []
        excluded = set()
        x = 1
        while len(result) < n:
            if x not in excluded:
                result.append(x)
                excluded.add(k - x)
            x += 1
        return sum(result)