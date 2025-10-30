class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i, (a, b, c, m) in enumerate(variables):
            if pow(pow(a, b, 10), c, m) == target:
                result.append(i)
        return result