import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx +=1

    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a = int(data[idx])
        idx +=1
        b = int(data[idx])
        idx +=1
        adj[a].append(b)
        adj[b].append(a)
    
    selected = list(map(int, data[idx:idx+K]))
    selected_set = set(selected)
    idx += K

    # Build parent and children structure
    root = selected[0]
    parent = [0] * (N+1)
    children = [[] for _ in range(N+1)]
    visited = [False] * (N+1)
    queue = deque([root])
    visited[root] = True

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                children[u].append(v)
                queue.append(v)
    
    # Compute subtree counts
    count = [0] * (N+1)
    stack = [(root, False)]
    while stack:
        node, visited_flag = stack.pop()
        if not visited_flag:
            stack.append((node, True))
            # Push children in reversed order to process in order
            for child in reversed(children[node]):
                stack.append((child, False))
        else:
            cnt = 0
            if node in selected_set:
                cnt += 1
            for child in children[node]:
                cnt += count[child]
            count[node] = cnt
    
    # Count edges
    sum_edges = 0
    K_val = K
    lower = 1
    upper = K_val -1
    for v in range(1, N+1):
        if v == root:
            continue
        if count[v] >= lower and count[v] <= upper:
            sum_edges +=1
    
    print(sum_edges + 1)

if __name__ == "__main__":
    main()