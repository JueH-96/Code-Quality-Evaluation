class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result = []
        i = 1
        while len(result) < n:
            if all(i + x != k for x in result):
                result.append(i)
            i += 1
        return sum(result)