class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i in range(len(variables)):
            a_i, b_i, c_i, m_i = variables[i]
            val = pow(a_i, b_i, 10)
            val = pow(val, c_i)
            val = val % m_i
            if val == target:
                good_indices.append(i)
        return good_indices