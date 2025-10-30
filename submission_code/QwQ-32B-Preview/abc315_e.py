import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    adj = [[] for _ in range(N+1)]
    reverse_adj = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        C = int(data[ptr])
        ptr += 1
        for _ in range(C):
            P = int(data[ptr])
            ptr += 1
            adj[P].append(i)
            reverse_adj[i].append(P)
    
    # BFS to find all prerequisites for book 1
    visited = set()
    queue = deque([1])
    while queue:
        k = queue.popleft()
        if k in visited:
            continue
        visited.add(k)
        for p in reverse_adj[k]:
            if p not in visited:
                queue.append(p)
    
    # Perform topological sort on visited books excluding book 1
    in_degree = [0] * (N + 1)
    for i in range(1, N+1):
        if i in visited:
            in_degree[i] = len(reverse_adj[i])
    in_degree[1] = 0  # Exclude book 1
    
    # Initialize queue with books that have in_degree 0, excluding book 1
    queue = deque()
    for i in visited:
        if in_degree[i] == 0 and i != 1:
            queue.append(i)
    
    result = []
    while queue:
        k = queue.popleft()
        result.append(str(k))
        for m in adj[k]:
            if m != 1:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    queue.append(m)
    
    print(' '.join(result))

if __name__ == '__main__':
    main()