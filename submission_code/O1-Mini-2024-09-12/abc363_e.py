# YOUR CODE HERE
import sys
import sys
import sys
def solve():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import defaultdict
    H, W, Y = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))
    cells = []
    for i in range(H):
        for j in range(W):
            cells.append((A[i][j], i, j))
    cells.sort()
    parent = [i for i in range(H * W +1)]
    size = [1]*(H * W +1)
    connected_to_sea = [False]*(H * W +1)
    sink_time = [0]*(H * W +1)
    sea = H * W
    freq = defaultdict(int)
    submerged = [False]*(H * W)
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if size[u_root] < size[v_root]:
            u_root, v_root = v_root, u_root
        parent[v_root] = u_root
        size[u_root] += size[v_root]
        if connected_to_sea[u_root] or connected_to_sea[v_root]:
            if not connected_to_sea[u_root] and connected_to_sea[v_root]:
                connected_to_sea[u_root] = True
                if sink_time[u_root] == 0:
                    sink_time[u_root] = current_A
                    freq[current_A] += size[u_root]
            elif connected_to_sea[u_root] and not connected_to_sea[v_root]:
                connected_to_sea[u_root] = True
                if sink_time[u_root] == 0:
                    sink_time[u_root] = current_A
                    freq[current_A] += size[u_root]
            elif connected_to_sea[u_root] and connected_to_sea[v_root]:
                # Both already connected, sink_time already set
                pass
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    idx = 0
    n = H * W
    while idx < n:
        current_A = cells[idx][0]
        # Process all cells with current_A
        same_A = []
        while idx < n and cells[idx][0] == current_A:
            same_A.append(cells[idx])
            idx +=1
        for (_, i,j) in same_A:
            pos = i * W + j
            submerged[pos] = True
            if i ==0 or i == H-1 or j ==0 or j == W-1:
                connected_to_sea[find(pos)] = True
            for di,dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    npos = ni * W + nj
                    if submerged[npos]:
                        union(pos, npos)
            root = find(pos)
            if connected_to_sea[root] and sink_time[root]==0:
                sink_time[root] = current_A
                freq[current_A] += size[root]
        # After processing all same_A cells, some unions may have happened
    # Now handle not connected to sea
    total = H * W
    connected = sum(freq.values())
    freq_inf = total - connected
    max_A = max([cell[0] for cell in cells] + [0])
    freq_array = [0]*(max(max_A, Y)+2)
    for t, cnt in freq.items():
        if t <= max_A:
            freq_array[t] += cnt
    if freq_inf >0:
        if max_A +1 <= max(max_A, Y)+1:
            freq_array[max(max_A, Y)+1] += freq_inf
        else:
            freq_array[max(max_A, Y)] += freq_inf
    prefix = [0]*(max(max_A, Y)+2)
    for t in range(1, len(prefix)):
        prefix[t] = prefix[t-1] + freq_array[t]
    for t in range(1, Y+1):
        if t < len(prefix):
            sunk = prefix[t]
        else:
            sunk = prefix[-1]
        remaining = total - sunk
        print(remaining)