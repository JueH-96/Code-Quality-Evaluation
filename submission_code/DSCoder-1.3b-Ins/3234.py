class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        n = len(variables)
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        prefix[0] = variables[0][0]
        suffix[n] = variables[n-1][0]
        for i in range(1,n):
            prefix[i] = (prefix[i-1]*variables[i][0])%10
        for i in range(n-2,-1,-1):
            suffix[i] = (suffix[i+1]*variables[i][0])%10
        res = []
        for i in range(1,n-1):
            if (prefix[i-1]**variables[i][1] % 10)**variables[i][2] == target and (suffix[i+1]**variables[i][1] % 10)**variables[i][2] == target:
                res.append(i)
        return res