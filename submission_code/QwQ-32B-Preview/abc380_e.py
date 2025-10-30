import sys

def main():
    import sys
    from sys import stdin

    # Read N and Q
    N_Q = stdin.readline().split()
    N = int(N_Q[0])
    Q = int(N_Q[1])

    # Initialize Union-Find data structures
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    size = [1] * (N + 1)
    color = list(range(N + 1))
    count = [1] * (N + 1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y, c):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return
        if size[root_x] < size[root_y]:
            parent[root_x] = root_y
            size[root_y] += size[root_x]
            color[root_y] = c
        else:
            parent[root_y] = root_x
            size[root_x] += size[root_y]
            color[root_x] = c

    for _ in range(Q):
        query = list(map(int, stdin.readline().split()))
        if query[0] == 1:
            x, c = query[1], query[2]
            root_x = find(x)
            old_c = color[root_x]
            count[old_c] -= size[root_x]
            color[root_x] = c
            count[c] += size[root_x]
            # Merge with left neighbor if it has color c
            if x > 1:
                neighbor = x - 1
                root_n = find(neighbor)
                if root_n != root_x and color[root_n] == c:
                    union(neighbor, x, c)
            # Merge with right neighbor if it has color c
            if x < N:
                neighbor = x + 1
                root_n = find(neighbor)
                if root_n != root_x and color[root_n] == c:
                    union(x, neighbor, c)
        elif query[0] == 2:
            c = query[1]
            print(count[c])

if __name__ == "__main__":
    main()