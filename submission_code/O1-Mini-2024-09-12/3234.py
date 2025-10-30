class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            first = pow(a, b, 10)
            result = pow(first, c, m)
            if result == target:
                good_indices.append(i)
        return good_indices