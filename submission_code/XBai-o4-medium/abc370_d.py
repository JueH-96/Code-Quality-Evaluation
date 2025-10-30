import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr]); ptr += 1
    W = int(input[ptr]); ptr += 1
    Q = int(input[ptr]); ptr += 1

    queries = []
    for _ in range(Q):
        R = int(input[ptr]); ptr += 1
        C = int(input[ptr]); ptr += 1
        queries.append((R, C))

    # Initialize DSUs
    max_H = H
    max_W = W

    # columns_up_parent: [c][i]
    columns_up_parent = [[0] * (max_H + 2) for _ in range(max_W + 2)]
    columns_down_parent = [[0] * (max_H + 2) for _ in range(max_W + 2)]
    # rows_left_parent: [r][j]
    rows_left_parent = [[0] * (max_W + 2) for _ in range(max_H + 2)]
    rows_right_parent = [[0] * (max_W + 2) for _ in range(max_H + 2)]

    # Initialize DSUs with each cell pointing to itself
    for c in range(1, W + 1):
        for i in range(1, H + 1):
            columns_up_parent[c][i] = i
            columns_down_parent[c][i] = i

    for r in range(1, H + 1):
        for j in range(1, W + 1):
            rows_left_parent[r][j] = j
            rows_right_parent[r][j] = j

    destroyed = [[False] * (W + 2) for _ in range(H + 2)]
    total_walls = H * W

    def find_up(c, i):
        original_i = i
        while True:
            if i < 1:
                return 0
            parent = columns_up_parent[c][i]
            if parent == i:
                break
            i = parent
        # Path compression
        while True:
            current = columns_up_parent[c][original_i]
            if current == i:
                break
            columns_up_parent[c][original_i] = i
            original_i = current
        return i

    def find_down(c, i):
        original_i = i
        while True:
            if i > H:
                return H + 1
            parent = columns_down_parent[c][i]
            if parent == i:
                break
            i = parent
        # Path compression
        while True:
            current = columns_down_parent[c][original_i]
            if current == i:
                break
            columns_down_parent[c][original_i] = i
            original_i = current
        return i

    def find_left(r, j):
        original_j = j
        while True:
            if j < 1:
                return 0
            parent = rows_left_parent[r][j]
            if parent == j:
                break
            j = parent
        # Path compression
        while True:
            current = rows_left_parent[r][original_j]
            if current == j:
                break
            rows_left_parent[r][original_j] = j
            original_j = current
        return j

    def find_right(r, j):
        original_j = j
        while True:
            if j > W:
                return W + 1
            parent = rows_right_parent[r][j]
            if parent == j:
                break
            j = parent
        # Path compression
        while True:
            current = rows_right_parent[r][original_j]
            if current == j:
                break
            rows_right_parent[r][original_j] = j
            original_j = current
        return j

    for (R, C) in queries:
        if not destroyed[R][C]:
            destroyed[R][C] = True
            total_walls -= 1

            # Update column up DSU
            root_up = find_up(C, R - 1)
            columns_up_parent[C][R] = root_up

            # Update column down DSU
            root_down = find_down(C, R + 1)
            columns_down_parent[C][R] = root_down

            # Update row left DSU
            root_left = find_left(R, C - 1)
            rows_left_parent[R][C] = root_left

            # Update row right DSU
            root_right = find_right(R, C + 1)
            rows_right_parent[R][C] = root_right
        else:
            to_destroy = []
            # Check up
            i_up = find_up(C, R - 1)
            if 1 <= i_up < R:
                to_destroy.append((i_up, C))
            # Check down
            i_down = find_down(C, R + 1)
            if 1 <= i_down <= H and i_down > R:
                to_destroy.append((i_down, C))
            # Check left
            j_left = find_left(R, C - 1)
            if 1 <= j_left < C:
                to_destroy.append((R, j_left))
            # Check right
            j_right = find_right(R, C + 1)
            if 1 <= j_right <= W and j_right > C:
                to_destroy.append((R, j_right))

            for (x, y) in to_destroy:
                if not destroyed[x][y]:
                    destroyed[x][y] = True
                    total_walls -= 1

                    # Update column up DSU for y, x
                    root_up = find_up(y, x - 1)
                    columns_up_parent[y][x] = root_up

                    # Update column down DSU
                    root_down = find_down(y, x + 1)
                    columns_down_parent[y][x] = root_down

                    # Update row left DSU for x, y
                    root_left = find_left(x, y - 1)
                    rows_left_parent[x][y] = root_left

                    # Update row right DSU
                    root_right = find_right(x, y + 1)
                    rows_right_parent[x][y] = root_right

    print(total_walls)

if __name__ == "__main__":
    main()