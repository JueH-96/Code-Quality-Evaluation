import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    from collections import defaultdict
    import heapq

    input = sys.stdin.readline
    N = int(input())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a,b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)

    removed = [False] * (N+1)
    size = [0] * (N+1)
    pairs = []

    # compute subtree sizes
    def dfs_size(u, p):
        size[u] = 1
        for v in adj[u]:
            if v != p and not removed[v]:
                dfs_size(v, u)
                size[u] += size[v]

    # find centroid
    def dfs_centroid(u, p, comp_size):
        for v in adj[u]:
            if v != p and not removed[v]:
                if size[v] > comp_size // 2:
                    return dfs_centroid(v, u, comp_size)
        return u

    # collect all nodes in this subtree
    def dfs_collect(u, p, lst):
        lst.append(u)
        for v in adj[u]:
            if v != p and not removed[v]:
                dfs_collect(v, u, lst)

    def solve(root):
        # compute sizes for component
        dfs_size(root, 0)
        comp_size = size[root]
        c = dfs_centroid(root, 0, comp_size)
        removed[c] = True
        # prepare lists: list 0 is centroid itself
        list_nodes = []
        list_nodes.append([c])
        # collect each child subtree
        for v in adj[c]:
            if not removed[v]:
                lst = []
                dfs_collect(v, c, lst)
                list_nodes.append(lst)

        # max-heap by size
        heap = []
        for idx, lst in enumerate(list_nodes):
            if lst:
                heapq.heappush(heap, (-len(lst), idx))

        # greedy pair across different lists
        while len(heap) >= 2:
            na, a = heapq.heappop(heap)
            nb, b = heapq.heappop(heap)
            # pop one node from each
            ua = list_nodes[a].pop()
            ub = list_nodes[b].pop()
            pairs.append((ua, ub))
            # push back if non-empty
            if list_nodes[a]:
                heapq.heappush(heap, (-(len(list_nodes[a])), a))
            if list_nodes[b]:
                heapq.heappush(heap, (-(len(list_nodes[b])), b))

        # if one list remains, process it recursively
        if heap:
            _, idx = heap[0]
            # idx cannot be 0 realistically (centroid), but handle generally
            # remaining nodes in list_nodes[idx] form one subtree component
            lst = list_nodes[idx]
            # all other lists are empty
            # recurse on that subtree
            # pick any node in that remaining list as new root
            solve(lst[0])
        # done for this component

    # start decomposition from node 1
    solve(1)

    # output pairs: total should be N/2
    out = sys.stdout.write
    for x,y in pairs:
        out(f"{x} {y}
")

if __name__ == "__main__":
    main()