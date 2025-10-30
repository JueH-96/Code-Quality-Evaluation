class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for index, (a, b, c, m) in enumerate(variables):
            first_step = pow(a, b, 10)
            second_step = pow(first_step, c, m)
            if second_step == target:
                result.append(index)
        return result