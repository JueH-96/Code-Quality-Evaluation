class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def mark(node, parent, phase):
            if phase >= ans[node]: return
            ans[node] = phase
            for neighbor in graph[node]:
                if neighbor != parent:
                    mark(neighbor, node, phase + 2 if phase % 2 else phase + 1)
        
        ans = [0] * n
        for i in range(n):
            ans[i] = math.inf
            mark(i, -1, 0)
        return ans