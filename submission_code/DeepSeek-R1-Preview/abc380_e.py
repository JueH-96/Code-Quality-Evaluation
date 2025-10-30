import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    # Initialize DSU data structures
    parent = list(range(N + 2))  # 0 to N+1, but cells are 1-based
    rank = [1] * (N + 2)
    color = list(range(N + 2))  # color of the set's root
    size = [1] * (N + 2)       # size of the set
    L = list(range(N + 2))     # leftmost cell of the set
    R = list(range(N + 2))     # rightmost cell of the set

    color_counts = [0] * (N + 2)
    for i in range(1, N + 1):
        color_counts[i] = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression
            x = parent[x]
        return x

    for _ in range(Q):
        t = input[ptr]
        ptr += 1
        if t == '1':
            x = int(input[ptr])
            ptr += 1
            c = int(input[ptr])
            ptr += 1

            root_x = find(x)
            if color[root_x] == c:
                continue

            old_color = color[root_x]
            color_counts[old_color] -= size[root_x]
            color_counts[c] += size[root_x]
            color[root_x] = c

            # Check left
            current_L = L[root_x]
            while current_L > 1:
                left_neighbor = current_L - 1
                root_left = find(left_neighbor)
                if color[root_left] != c:
                    break
                if root_left == root_x:
                    current_L = L[root_x]
                    continue

                # Union: merge root_left into root_x
                if rank[root_x] < rank[root_left]:
                    root_x, root_left = root_left, root_x

                parent[root_left] = root_x
                rank[root_x] += rank[root_left]
                size[root_x] += size[root_left]
                L[root_x] = min(L[root_x], L[root_left])
                R[root_x] = max(R[root_x], R[root_left])
                current_L = L[root_x]

            # Check right
            current_R = R[root_x]
            while current_R < N:
                right_neighbor = current_R + 1
                root_right = find(right_neighbor)
                if color[root_right] != c:
                    break
                if root_right == root_x:
                    current_R = R[root_x]
                    continue

                if rank[root_x] < rank[root_right]:
                    root_x, root_right = root_right, root_x

                parent[root_right] = root_x
                rank[root_x] += rank[root_right]
                size[root_x] += size[root_right]
                R[root_x] = max(R[root_x], R[root_right])
                L[root_x] = min(L[root_x], L[root_right])
                current_R = R[root_x]

        else:
            c = int(input[ptr])
            ptr += 1
            print(color_counts[c])

if __name__ == '__main__':
    main()