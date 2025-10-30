class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        graph = defaultdict(list)
        for i, r in enumerate(receiver):
            graph[i].append(r)
        
        def dfs(node, k):
            if k == 0:
                return node
            return dfs(graph[node][0], k-1)
        
        max_val = 0
        for i in range(n):
            curr_val = i
            curr_node = i
            for _ in range(k):
                curr_node = graph[curr_node][0]
                curr_val += curr_node
            max_val = max(max_val, curr_val)
        
        return max_val