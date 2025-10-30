class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for index, var in enumerate(variables):
            a, b, c, m = var
            temp = pow(a, b, 10)
            final_result = pow(temp, c, m)
            if final_result == target:
                good_indices.append(index)
        return good_indices