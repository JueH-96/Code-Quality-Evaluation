class Solution:
    def findAnswer(self, parent, s):
        children = [[] for _ in parent]
        for i, p in enumerate(parent): 
            if ~p: children[p].append(i)
        
        def dfs(i): 
            x = s[i]
            for j in sorted(children[i]): x += dfs(j)
            return x
        
        ans = [False] * len(parent)
        for i in range(len(parent)): ans[i] = dfs(i) == dfs(i)[::-1]
        return ans