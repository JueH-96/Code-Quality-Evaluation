import sys
import heapq
from collections import deque, defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    tree = defaultdict(list)
    for _ in range(N-1):
        u = int(data[idx])
        v = int(data[idx+1])
        l = int(data[idx+2])
        tree[u].append((v, l))
        tree[v].append((u, l))
        idx += 3

    parent = [0] * (N + 1)
    depth_sum = [0] * (N + 1)
    edge_to_parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v, l in tree[u]:
            if not visited[v] and v != 1:
                visited[v] = True
                parent[v] = u
                edge_to_parent[v] = l
                depth_sum[v] = depth_sum[u] + l
                q.append(v)

    heap = []
    for i in range(1, N+1):
        if i != 1:
            heapq.heappush(heap, (-depth_sum[i], i))

    covered = [False] * (N + 1)
    res = [0] * (N + 1)
    total_sum = 0

    for k in range(1, N+1):
        while True:
            if not heap:
                break
            neg_ds, u = heapq.heappop(heap)
            current_ds = -neg_ds
            if current_ds > 0:
                break
        if not heap:
            break
        current = u
        add = 0
        while current != 1:
            if not covered[current]:
                add += edge_to_parent[current]
                covered[current] = True
            current = parent[current]
        total_sum += add
        res[k] = total_sum

    for k in range(1, N+1):
        print(2 * res[k])

if __name__ == "__main__":
    main()