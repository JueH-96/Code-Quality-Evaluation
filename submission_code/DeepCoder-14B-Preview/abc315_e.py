from collections import deque
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    
    prereqs = [[] for _ in range(n + 1)]  # 1-based indexing
    
    for i in range(1, n + 1):
        c_i = int(input[ptr])
        ptr += 1
        if c_i > 0:
            prereqs[i] = list(map(int, input[ptr:ptr + c_i]))
            ptr += c_i
        else:
            prereqs[i] = []
    
    # Collect all books that must be read before book 1
    s = set()
    q = deque()
    
    for p in prereqs[1]:
        if p not in s:
            s.add(p)
            q.append(p)
    
    while q:
        current = q.popleft()
        for p in prereqs[current]:
            if p not in s:
                s.add(p)
                q.append(p)
    
    # If no books needed, but the problem says C_1 >=1, so this won't happen
    if not s:
        print()
        return
    
    # Build the adjacency list and in-degree
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    
    for i in s:
        for p in prereqs[i]:
            if p in s:
                adj[p].append(i)
                in_degree[i] += 1
    
    # Initialize queue with nodes having in_degree zero
    queue = deque()
    for node in s:
        if in_degree[node] == 0:
            queue.append(node)
    
    top_order = []
    
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Output the topological order
    print(' '.join(map(str, top_order)))

if __name__ == '__main__':
    main()