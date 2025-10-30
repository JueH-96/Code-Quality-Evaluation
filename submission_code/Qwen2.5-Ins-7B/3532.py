from collections import defaultdict

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def dfs(node, parent, time):
            times = [time]
            for neighbor in graph[node]:
                if neighbor != parent:
                    times.append(dfs(neighbor, node, time + 1))
            return max(times)
        
        def calculate_times(node, parent, even, odd):
            if node % 2 == 0:
                return max(even + 1, odd + 2)
            else:
                return max(even + 2, odd + 1)
        
        def find_max_times(node, parent, even, odd):
            max_even, max_odd = 0, 0
            for neighbor in graph[node]:
                if neighbor != parent:
                    new_even, new_odd = find_max_times(neighbor, node, even, odd)
                    max_even = max(max_even, new_even)
                    max_odd = max(max_odd, new_odd)
            return max_even + 1, max_odd + 1
        
        max_even, max_odd = find_max_times(0, -1, 0, 0)
        result = [0] * len(edges) + [0]
        
        for i in range(len(edges) - 1, -1, -1):
            u, v = edges[i]
            if result[u] == 0:
                result[u] = calculate_times(u, v, max_even, max_odd)
            if result[v] == 0:
                result[v] = calculate_times(v, u, max_even, max_odd)
        
        return result