# YOUR CODE HERE
import sys
import threading

sys.setrecursionlimit(1 << 25)
def main():
    import sys
    input = sys.stdin.readline

    N = int(sys.stdin.readline())
    a_list = list(map(int, sys.stdin.readline().split()))
    a = [0] + a_list  # 1-indexed

    graph = [[] for _ in range(N+1)]
    for u in range(1, N+1):
        v = a[u]
        graph[u].append(v)

    index = 0
    index_stack = []
    on_stack = [False] * (N+1)
    indices = [0] * (N+1)
    lowlink = [0] * (N+1)
    scc_id = [0] * (N+1)
    scc_sizes = []
    scc_count = 0

    def strongconnect(u):
        nonlocal index, scc_count
        index += 1
        indices[u] = index
        lowlink[u] = index
        index_stack.append(u)
        on_stack[u] = True

        for v in graph[u]:
            if indices[v] == 0:
                strongconnect(v)
                lowlink[u] = min(lowlink[u], lowlink[v])
            elif on_stack[v]:
                lowlink[u] = min(lowlink[u], indices[v])

        if lowlink[u] == indices[u]:
            scc = []
            while True:
                v = index_stack.pop()
                on_stack[v] = False
                scc_id[v] = scc_count
                scc.append(v)
                if v == u:
                    break
            scc_sizes.append(len(scc))
            scc_count +=1

    for u in range(1, N+1):
        if indices[u] == 0:
            strongconnect(u)

    reach_size = [-1] * (N+1)
    size_scc = [0] * scc_count
    for i in range(scc_count):
        size_scc[i] = scc_sizes[i]

    def dfs(u):
        if reach_size[u] != -1:
            return reach_size[u]
        scc_u = scc_id[u]
        if size_scc[scc_u] > 1:
            reach_size[u] = size_scc[scc_u]
            return reach_size[u]
        v = a[u]
        if u == v:
            # Self-loop
            reach_size[u] = 1
            return reach_size[u]
        else:
            reach_size[u] = 0  # Mark as visited to avoid cycles (shouldn't be any)
            reach_size[u] = 1 + dfs(v)
            return reach_size[u]

    total_pairs = 0
    for u in range(1, N+1):
        total_pairs += dfs(u)

    print(total_pairs)

threading.Thread(target=main).start()