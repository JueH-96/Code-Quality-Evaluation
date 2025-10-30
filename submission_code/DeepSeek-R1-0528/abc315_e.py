import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    dependencies = [[] for _ in range(n+1)]
    index = 1
    for i in range(1, n+1):
        c = int(data[index]); index += 1
        deps = []
        for j in range(c):
            deps.append(int(data[index])); index += 1
        dependencies[i] = deps
    
    required = set()
    stack = [1]
    while stack:
        book = stack.pop()
        for prereq in dependencies[book]:
            if prereq not in required:
                required.add(prereq)
                stack.append(prereq)
    
    graph = [[] for _ in range(n+1)]
    in_degree = [0] * (n+1)
    
    for node in required:
        for prereq in dependencies[node]:
            if prereq in required:
                graph[prereq].append(node)
                in_degree[node] += 1
                
    zero_degree_nodes = []
    for node in required:
        if in_degree[node] == 0:
            zero_degree_nodes.append(node)
            
    zero_degree_nodes.sort(reverse=True)
    q = deque(zero_degree_nodes)
    order = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                q.append(nxt)
                
    print(" ".join(map(str, order)))

if __name__ == "__main__":
    main()