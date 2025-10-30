import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    edges = []
    for i in range(M):
        u = int(data[2 + i * 2]) - 1
        v = int(data[3 + i * 2]) - 1
        edges.append((u, v))
    
    W = list(map(int, data[2 + 2 * M:2 + 2 * M + N]))
    A = list(map(int, data[2 + 2 * M + N:]))
    
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    def can_place(x, S):
        return sum(W[y] for y in S) < W[x]
    
    def max_operations():
        total_operations = 0
        queue = deque()
        
        for i in range(N):
            if A[i] > 0:
                queue.append(i)
        
        while queue:
            x = queue.popleft()
            if A[x] > 0:
                A[x] -= 1
                total_operations += 1
                S = [y for y in graph[x] if can_place(x, [y])]
                for y in S:
                    A[y] += 1
                    queue.append(y)
        
        return total_operations
    
    print(max_operations())

if __name__ == "__main__":
    solve()