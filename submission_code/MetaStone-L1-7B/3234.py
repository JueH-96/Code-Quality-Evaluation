class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i in range(len(variables)):
            a, b, c, m = variables[i]
            x = pow(a, b, 10)
            y = pow(x, c, m)
            if y == target:
                good_indices.append(i)
        return good_indices