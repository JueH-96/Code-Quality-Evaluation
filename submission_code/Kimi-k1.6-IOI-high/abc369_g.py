from sys import stdin
import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    adj = [[] for _ in range(N+1)]  # 1-based
    parent = [0]*(N+1)
    depth = [0]*(N+1)
    children = [[] for _ in range(N+1)]
    root = 1  # assuming root is 1

    for _ in range(N-1):
        u = int(data[idx])
        v = int(data[idx+1])
        l = int(data[idx+2])
        idx +=3
        adj[u].append((v, l))
        adj[v].append((u, l))
    
    # BFS to compute parent and depth
    from collections import deque
    q = deque()
    q.append(root)
    visited = [False]*(N+1)
    visited[root] = True
    while q:
        u = q.popleft()
        for v, l in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + l
                children[u].append((v, l))
                q.append(v)
    
    # Compute max depth in subtree (post-order DFS)
    max_depth = [0]*(N+1)
    def dfs(u):
        current_max = 0
        for v, l in children[u]:
            dfs(v)
            current_max = max(current_max, max_depth[v])
        max_depth[u] = depth[u] + current_max
    dfs(root)
    
    # Prepare priority queue entries
    # Use a max-heap, but Python has a min-heap, so invert values
    heap = []
    used = [False]*(N+1)
    heapq.heappush(heap, (-max_depth[root], root, 0))  # (-value, node, depth to contribute)
    
    ans = [0]*(N+1)
    current_sum = 0
    selected = 0
    while heap and selected < N:
        val_neg, u, contrib = heapq.heappop(heap)
        val = -val_neg
        if used[u]:
            continue
        # Check if parent is used, if so, this path is covered
        ancestors = []
        while u != root and not used[u]:
            ancestors.append(u)
            u = parent[u]
        if used[u]:
            # An ancestor is used, process children
            for node in ancestors:
                used[node] = True
            continue
        # Now u is root, not used, process all ancestors
        for node in ancestors:
            used[node] = True
        current_sum += contrib
        selected +=1
        if selected > N:
            break
        ans[selected] = current_sum *2
        # Add children of the selected node's parent (if any)
        # Wait, the selected node's children's own children will contribute based on their max_depth minus their depth
        # Wait, after selecting u, any node in its subtree is now considered as starting from u
        for v, l in children[u]:
            heapq.heappush(heap, (-max_depth[v], v, l))
    
    for i in range(1, N+1):
        if i <= len(ans)-1 and ans[i] !=0:
            print(ans[i])
        else:
            print(current_sum *2)

if __name__ == '__main__':
    main()