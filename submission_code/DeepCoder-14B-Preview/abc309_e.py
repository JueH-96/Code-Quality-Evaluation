from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1
    
    parents = list(map(int, data[idx:idx + n-1]))
    idx += n-1
    
    children = [[] for _ in range(n+1)]  # 1-based indexing
    for i in range(2, n+1):
        p = parents[i-2]
        children[p].append(i)
    
    max_depth = [-1] * (n+1)
    for _ in range(m):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        if max_depth[x] < y:
            max_depth[x] = y
    
    count = 0
    queue = deque()
    queue.append((1, max_depth[1]))
    
    visited = [False] * (n+1)
    visited[1] = True
    
    while queue:
        u, allowed = queue.popleft()
        if allowed >= 0:
            count += 1
        for v in children[u]:
            if not visited[v]:
                visited[v] = True
                v_allowed = max_depth[v]
                if allowed >= 0:
                    v_allowed = max(v_allowed, allowed - 1)
                queue.append((v, v_allowed))
    
    print(count)

if __name__ == "__main__":
    main()