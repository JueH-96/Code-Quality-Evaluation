class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i, var in enumerate(variables):
            a, b, c, m = var
            temp1 = pow(a, b, 10)
            temp2 = pow(temp1, c, m)
            if temp2 == target:
                result.append(i)
        return result