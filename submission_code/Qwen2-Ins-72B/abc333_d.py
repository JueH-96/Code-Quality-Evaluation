from collections import defaultdict, deque

def main():
    N = int(input())
    edges = [list(map(int, input().split())) for _ in range(N-1)]
    
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    queue = deque()
    leaf_count = 0
    for node in graph:
        if len(graph[node]) == 1:
            queue.append(node)
            leaf_count += 1
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            graph[neighbor].remove(node)
            if len(graph[neighbor]) == 1 and neighbor != 1:
                queue.append(neighbor)
                leaf_count += 1
    
    print(N - leaf_count - 1)

main()