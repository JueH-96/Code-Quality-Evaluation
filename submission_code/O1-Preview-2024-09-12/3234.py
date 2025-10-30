class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        ans = []
        for i, (a_i, b_i, c_i, m_i) in enumerate(variables):
            x = pow(a_i, b_i, 10)
            result = pow(x, c_i, m_i)
            if result == target:
                ans.append(i)
        return ans