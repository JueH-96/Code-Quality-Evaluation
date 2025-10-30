from collections import defaultdict, deque
import sys
input = sys.stdin.readline

class Solution:
    def solve(self):
        N = int(input())
        edges = [list(map(int, input().split())) for _ in range(N-1)]
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Find the center of the tree
        queue = deque([1])
        visited = [False] * (N+1)
        visited[1] = True
        while queue:
            node = queue.popleft()
            if all(visited[neighbor] for neighbor in graph[node]):
                center = node
                break
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
        
        # Calculate the levels of the stars
        levels = []
        visited = [False] * (N+1)
        visited[center] = True
        for neighbor in graph[center]:
            queue = deque([neighbor])
            visited[neighbor] = True
            level = 1
            while queue:
                for _ in range(len(queue)):
                    node = queue.popleft()
                    if len(graph[node]) == 1:
                        levels.append(level)
                    for next_node in graph[node]:
                        if not visited[next_node]:
                            queue.append(next_node)
                            visited[next_node] = True
                level += 1
        
        print(*sorted(levels))
        
Solution().solve()