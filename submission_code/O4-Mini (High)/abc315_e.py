import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    # adj[i] = list of books you must read before i (prerequisites)
    adj = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        data = list(map(int, input().split()))
        c = data[0]
        if c:
            adj[i] = data[1:]
    # We do a DFS on the reversed-dependency graph (i -> prerequisites)
    # starting from book 1, collect all reachable books and output them
    # in post-order.  That guarantees any book's prerequisites come
    # earlier in the list.
    visited = [False] * (N+1)
    visited[1] = True
    stack = [[1, 0]]  # each entry [node, next_child_index]
    res = []
    while stack:
        node, idx = stack[-1]
        if idx < len(adj[node]):
            nei = adj[node][idx]
            stack[-1][1] = idx + 1
            if not visited[nei]:
                visited[nei] = True
                stack.append([nei, 0])
        else:
            # all children processed -> post-order append
            stack.pop()
            res.append(node)
    # res is the post-order on the reversed graph; the last entry is 1
    res.pop()  # remove book 1 itself
    # print the rest in the order they finished
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()