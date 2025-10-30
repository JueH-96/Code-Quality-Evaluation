def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster I/O if needed:
    # import sys
    # data = sys.stdin.read().strip().split()
    # Then iterate over data with an index

    # DSU (Union-Find) with an additional structure to keep track
    # of up to 10 largest elements in each component.
    sys.setrecursionlimit(10**7)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        rootA = find(a)
        rootB = find(b)
        if rootA == rootB:
            return
        # Union by size
        if size[rootA] < size[rootB]:
            rootA, rootB = rootB, rootA
        parent[rootB] = rootA
        size[rootA] += size[rootB]
        # Merge the top10 lists
        merged = []
        ia, ib = 0, 0
        la, lb = len(top10[rootA]), len(top10[rootB])
        # Both sorted descending
        while (ia < la or ib < lb) and len(merged) < 10:
            valA = top10[rootA][ia] if ia < la else -1
            valB = top10[rootB][ib] if ib < lb else -1
            if valA >= valB:
                merged.append(valA)
                ia += 1
            else:
                merged.append(valB)
                ib += 1
        top10[rootA] = merged
        # Clear the merged-from list to save memory
        top10[rootB] = []

    idx = 0
    N = int(input_data[idx]); idx += 1
    Q = int(input_data[idx]); idx += 1

    parent = list(range(N+1))
    size = [1]*(N+1)
    # Each node starts with itself in the top10 list.
    # We'll keep them sorted in descending order.
    top10 = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        top10[i].append(i)

    out = []
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        if t == 1:
            u = int(input_data[idx]); idx += 1
            v = int(input_data[idx]); idx += 1
            union(u, v)
        else:
            v = int(input_data[idx]); idx += 1
            k = int(input_data[idx]); idx += 1
            r = find(v)
            if len(top10[r]) < k:
                out.append("-1")
            else:
                out.append(str(top10[r][k-1]))

    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()