class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result = []
        exclude = set()
        x = 1
        while len(result) < n:
            if x not in exclude:
                result.append(x)
                exclude.add(k - x)
            x += 1
        return sum(result)