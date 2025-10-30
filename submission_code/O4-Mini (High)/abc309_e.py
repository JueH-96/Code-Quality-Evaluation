import sys
import threading

def main():
    import sys
    from collections import deque

    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0]); M = int(data[1])

    # Read parent array and build children adjacency list
    p_list = list(map(int, sys.stdin.readline().split()))
    children = [[] for _ in range(N+1)]
    for i, p in enumerate(p_list, start=2):
        children[p].append(i)

    # Compute depth of each node by BFS
    depth = [0] * (N+1)
    dq = deque([1])
    depth[1] = 0
    while dq:
        u = dq.popleft()
        for v in children[u]:
            depth[v] = depth[u] + 1
            dq.append(v)

    # Read queries and compute L = depth[x] + y, group by x
    queries_by_node = [[] for _ in range(N+1)]
    for _ in range(M):
        line = sys.stdin.readline().split()
        x = int(line[0]); y = int(line[1])
        L = depth[x] + y
        queries_by_node[x].append(L)

    # We will do a DFS, maintaining a max-heap of active L-values
    import heapq
    heap = []
    removed_count = {}
    covered_count = 0

    # Iterative DFS stack: (node, state)
    # state 0 = enter, state 1 = exit
    stack = [(1, 0)]
    while stack:
        u, state = stack.pop()
        if state == 0:
            # ENTER u
            # schedule exit after children
            stack.append((u, 1))
            # push all queries at u
            for L in queries_by_node[u]:
                heapq.heappush(heap, -L)
            # clean up any stale (removed) heap-tops
            while heap and removed_count.get(heap[0], 0) > 0:
                v = heap[0]
                removed_count[v] -= 1
                if removed_count[v] == 0:
                    del removed_count[v]
                heapq.heappop(heap)
            # check coverage: max active L >= depth[u] ?
            if heap and -heap[0] >= depth[u]:
                covered_count += 1
            # push children for DFS (in reverse so we visit in original order)
            for v in reversed(children[u]):
                stack.append((v, 0))
        else:
            # EXIT u: mark removal of queries at u
            for L in queries_by_node[u]:
                v = -L
                removed_count[v] = removed_count.get(v, 0) + 1
            # actual heap pop of these removals is deferred to next cleanup

    # Output answer
    print(covered_count)

if __name__ == "__main__":
    main()