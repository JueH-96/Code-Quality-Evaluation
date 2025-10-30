class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        ans = []
        for i, var in enumerate(variables):
            a, b, c, m = var
            calc = (pow(a,b,10)**c) % m
            if calc == target:
                ans.append(i)
        return ans