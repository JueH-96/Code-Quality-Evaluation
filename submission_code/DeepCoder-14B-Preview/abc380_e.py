def find(u):
    while parent[u] != u:
        parent[u] = parent[parent[u]]
        u = parent[u]
    return u

def union(a, b):
    root_a = find(a)
    root_b = find(b)
    if root_a == root_b:
        return
    if size[root_a] < size[root_b]:
        root_a, root_b = root_b, root_a
    parent[root_b] = root_a
    left[root_a] = min(left[root_a], left[root_b])
    right[root_a] = max(right[root_a], right[root_b])
    size[root_a] += size[root_b]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    parent = list(range(N + 2))  # 0 to N+1
    color = list(range(N + 2))    # color[i] is i for each cell i
    left = list(range(N + 2))
    right = list(range(N + 2))
    size = [1] * (N + 2)
    color_count = [0] * (N + 2)

    for i in range(1, N + 1):
        color_count[i] += 1

    for _ in range(Q):
        query_type = data[ptr]
        ptr += 1
        if query_type == '1':
            x = int(data[ptr])
            ptr += 1
            c = int(data[ptr])
            ptr += 1
            root_x = find(x)
            if root_x == 0:
                continue
            if color[root_x] == c:
                continue
            # Update color_count
            old_color = color[root_x]
            color_count[old_color] -= size[root_x]
            color_count[c] += size[root_x]
            # Change root_x's color to c
            color[root_x] = c
            # Now, check left and right for possible merges
            current_root = root_x
            while True:
                # Check left
                if current_root != 0 and left[current_root] > 1:
                    cell = left[current_root] - 1
                    root_left = find(cell)
                    if color[root_left] == c:
                        # Subtract root_left's size from its old color
                        old_left_color = color[root_left]
                        color_count[old_left_color] -= size[root_left]
                        # Add to c
                        color_count[c] += size[root_left]
                        # Merge
                        union(current_root, root_left)
                        # Update current_root to the new root
                        current_root = find(left[current_root])
                        # Continue the loop
                        continue
                # Check right
                if current_root != 0 and right[current_root] < N:
                    cell = right[current_root] + 1
                    root_right = find(cell)
                    if color[root_right] == c:
                        # Subtract root_right's size from its old color
                        old_right_color = color[root_right]
                        color_count[old_right_color] -= size[root_right]
                        # Add to c
                        color_count[c] += size[root_right]
                        # Merge
                        union(current_root, root_right)
                        # Update current_root to the new root
                        current_root = find(right[current_root])
                        # Continue the loop
                        continue
                # If no more merges on either side, break
                break
        else:
            c = int(data[ptr])
            ptr += 1
            print(color_count[c])

if __name__ == "__main__":
    main()