import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1

    parent = list(range(N+1))  # 1-based
    offset = [(0,0)] * (N+1)  # offset to parent

    def find(u):
        original_u = u
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        root = u
        # Compute cumulative offsets
        cumul Off = []
        current_offset = (0, 0)
        for v in reversed(path):
            current_offset = offset[v][0] + current_offset[0], offset[v][1] + current_offset[1]
            cumul Off.append(current_offset)
        cumul Off = list(reversed(cumul Off))
        for i in range(len(path)):
            v = path[i]
            parent[v] = root
            offset[v] = cumul Off[i]
        return (root, offset[original_u][0], offset[original_u][1])

    for _ in range(M):
        A = int(input[idx]); idx +=1
        B = int(input[idx]); idx +=1
        X = int(input[idx]); idx +=1
        Y = int(input[idx]); idx +=1

        if A == 1:
            rootB, dxB, dyB = find(B)
            parent[rootB] = 1
            offset[rootB] = (X - dxB, Y - dyB)
        elif B == 1:
            rootA, dxA, dyA = find(A)
            parent[rootA] = 1
            offset[rootA] = (-X - dxA, -Y - dyA)
        else:
            rootA, dxA, dyA = find(A)
            rootB, dxB, dyB = find(B)
            if rootA != rootB:
                parent[rootB] = rootA
                ox = dxA + X - dxB
                oy = dyA + Y - dyB
                offset[rootB] = (ox, oy)

    for i in range(1, N+1):
        root, dx, dy = find(i)
        if root != 1:
            print("undecidable")
        else:
            print(dx, dy)

if __name__ == "__main__":
    main()