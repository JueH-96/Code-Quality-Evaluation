import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    
    parent = list(range(N + 2))  # 1-based indexing
    rank = [1] * (N + 2)
    min_idx = list(range(N + 2))
    max_idx = list(range(N + 2))
    color = list(range(N + 2))
    size = [1] * (N + 2)
    count = [0] * (N + 2)
    for i in range(1, N + 1):
        count[i] = 1
    
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
        if rank[u_root] < rank[v_root]:
            u_root, v_root = v_root, u_root
        parent[v_root] = u_root
        if rank[u_root] == rank[v_root]:
            rank[u_root] += 1
        min_idx[u_root] = min(min_idx[u_root], min_idx[v_root])
        max_idx[u_root] = max(max_idx[u_root], max_idx[v_root])
        size[u_root] += size[v_root]
    
    for _ in range(Q):
        query_type = data[ptr]
        ptr += 1
        if query_type == '1':
            x = int(data[ptr])
            ptr += 1
            c = int(data[ptr])
            ptr += 1
            rep_x = find(x)
            old_color = color[rep_x]
            if old_color == c:
                continue
            count[old_color] -= size[rep_x]
            count[c] += size[rep_x]
            color[rep_x] = c
            left_idx = min_idx[rep_x] - 1
            if left_idx >= 1:
                rep_left = find(left_idx)
                if color[rep_left] == c:
                    union(rep_x, rep_left)
            right_idx = max_idx[rep_x] + 1
            if right_idx <= N:
                rep_right = find(right_idx)
                if color[rep_right] == c:
                    union(rep_x, rep_right)
        else:
            c = int(data[ptr])
            ptr += 1
            print(count[c])

if __name__ == "__main__":
    main()