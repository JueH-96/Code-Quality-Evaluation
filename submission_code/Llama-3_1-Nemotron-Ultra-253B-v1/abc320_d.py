import sys

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    parent = list(range(N + 1))
    dx = [0] * (N + 1)
    dy = [0] * (N + 1)

    def find(u):
        if parent[u] == u:
            return (u, 0, 0)
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        # u is root now
        rev_path = list(reversed(path))
        cumulative_x = 0
        cumulative_y = 0
        offset_list = []
        for node in rev_path:
            cumulative_x += dx[node]
            cumulative_y += dy[node]
            offset_list.append((cumulative_x, cumulative_y))
        offset_list = offset_list[::-1]
        for i in range(len(path)):
            node = path[i]
            parent[node] = u
            dx[node] = offset_list[i][0]
            dy[node] = offset_list[i][1]
        if path:
            return (u, offset_list[0][0], offset_list[0][1])
        else:
            return (u, 0, 0)

    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        X = int(input[ptr])
        ptr += 1
        Y = int(input[ptr])
        ptr += 1
        rootA, dxA, dyA = find(A)
        rootB, dxB, dyB = find(B)
        if rootA != rootB:
            parent[rootB] = rootA
            dx[rootB] = dxA + X - dxB
            dy[rootB] = dyA + Y - dyB

    root1, dx1, dy1 = find(1)
    for u in range(1, N + 1):
        root_u, dxu, dyu = find(u)
        if root_u != root1:
            print("undecidable")
        else:
            print(dxu - dx1, dyu - dy1)

if __name__ == '__main__':
    main()