def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    index = 2
    p = list(map(int, data[index:index+N-1]))
    index += N-1
    
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        parent = p[i-2]
        children[parent].append(i)
    
    in_time = [0] * (N+1)
    out_time = [0] * (N+1)
    depth = [0] * (N+1)
    time = 0

    def dfs(u, d):
        nonlocal time
        time += 1
        in_time[u] = time
        depth[u] = d
        for v in children[u]:
            dfs(v, d+1)
        out_time[u] = time
    
    dfs(1, 0)
    
    insurances = []
    for _ in range(M):
        x_i = int(data[index])
        y_i = int(data[index+1])
        index += 2
        insurances.append((in_time[x_i], out_time[x_i], depth[x_i] + y_i))
    
    size = 1
    while size < N+2:
        size <<= 1
    
    tree = [0] * (2 * size)
    lazy = [0] * (2 * size)
    
    def push(node, node_left, node_right):
        if lazy[node] != 0:
            tree[node] = max(tree[node], lazy[node])
            if node_left != node_right:
                lazy[2*node] = max(lazy[2*node], lazy[node])
                lazy[2*node+1] = max(lazy[2*node+1], lazy[node])
            lazy[node] = 0
    
    def range_update(l, r, val):
        def _range_update(node, node_left, node_right, l, r, val):
            push(node, node_left, node_right)
            if r < node_left or node_right < l:
                return
            if l <= node_left and node_right <= r:
                lazy[node] = max(lazy[node], val)
                push(node, node_left, node_right)
                return
            mid = (node_left + node_right) // 2
            _range_update(2*node, node_left, mid, l, r, val)
            _range_update(2*node+1, mid+1, node_right, l, r, val)
            tree[node] = max(tree[2*node], tree[2*node+1])
        _range_update(1, 1, size, l, r, val)
    
    def point_query(pos):
        def _point_query(node, node_left, node_right, pos):
            push(node, node_left, node_right)
            if node_left == node_right:
                return tree[node]
            mid = (node_left + node_right) // 2
            if pos <= mid:
                return _point_query(2*node, node_left, mid, pos)
            else:
                return _point_query(2*node+1, mid+1, node_right, pos)
        return _point_query(1, 1, size, pos)
    
    for ins in insurances:
        l, r, val = ins
        range_update(l, r, val)
    
    covered = 0
    for u in range(1, N+1):
        max_depth = point_query(in_time[u])
        if depth[u] <= max_depth:
            covered += 1
    
    print(covered)

if __name__ == '__main__':
    main()