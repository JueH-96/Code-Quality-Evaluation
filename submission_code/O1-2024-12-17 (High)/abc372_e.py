def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    # parse N, Q
    N, Q = map(int, input_data[:2])
    
    # DSU (Disjoint Set Union) implementation
    parent = list(range(N+1))
    size = [1]*(N+1)
    
    # top10[r] will store the up to 10 largest vertices in the connected component
    # whose root is r. For non-root nodes, top10[node] will be None.
    top10 = [[i] for i in range(N+1)]
    
    # find function with path compression
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    # merge two top-10 lists (both sorted in descending order)
    def merge_top10(a, b):
        i = j = 0
        c = []
        while i < len(a) and j < len(b) and len(c) < 10:
            if a[i] > b[j]:
                c.append(a[i])
                i += 1
            else:
                c.append(b[j])
                j += 1
        while i < len(a) and len(c) < 10:
            c.append(a[i])
            i += 1
        while j < len(b) and len(c) < 10:
            c.append(b[j])
            j += 1
        return c
    
    # union by size
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        # merge top10
        merged = merge_top10(top10[rx], top10[ry])
        top10[rx] = merged
        top10[ry] = None
    
    # get the k-th largest vertex in the component of v
    def get_kth_largest(v, k):
        r = find(v)
        arr = top10[r]
        if arr is None or k > len(arr):
            return -1
        return arr[k-1]
    
    idx = 2  # index in input_data
    output = []
    for _ in range(Q):
        t = int(input_data[idx]); idx+=1
        if t == 1:
            # type 1 query: 1 u v
            u = int(input_data[idx]); idx+=1
            v = int(input_data[idx]); idx+=1
            union(u, v)
        else:
            # type 2 query: 2 v k
            v = int(input_data[idx]); idx+=1
            k = int(input_data[idx]); idx+=1
            output.append(str(get_kth_largest(v, k)))
    
    print("
".join(output))

# Do not forget to call main()
main()