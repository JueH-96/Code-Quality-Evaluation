import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    
    # Initialize data structures
    parent = list(range(N + 2))  # 1-based to N
    color = list(range(N + 2))
    left = list(range(N + 2))
    right = list(range(N + 2))
    size = [1] * (N + 2)
    count = [0] * (N + 2)
    for i in range(1, N + 1):
        count[i] = 1
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression
            x = parent[x]
        return x
    
    output = []
    for _ in range(Q):
        query_type = input[idx]
        idx += 1
        if query_type == '1':
            x = int(input[idx])
            idx += 1
            c = int(input[idx])
            idx += 1
            root = find(x)
            old_color = color[root]
            # Update the count for the old color and the new color
            count[old_color] -= size[root]
            color[root] = c
            count[c] += size[root]
            # Merge with left and right neighbors
            merged = True
            while merged:
                merged = False
                # Check left neighbor
                current_left = left[root]
                new_left = current_left - 1
                if new_left >= 1:
                    l_root = find(new_left)
                    if color[l_root] == c and right[l_root] == new_left:
                        # Merge l_root into root
                        parent[root] = l_root
                        right[l_root] = right[root]
                        size[l_root] += size[root]
                        # Update root to new root
                        root = l_root
                        merged = True
                # Check right neighbor if not merged yet or after left merge
                current_right = right[root]
                new_right = current_right + 1
                if new_right <= N:
                    r_root = find(new_right)
                    if color[r_root] == c and left[r_root] == new_right:
                        parent[root] = r_root
                        left[r_root] = left[root]
                        size[r_root] += size[root]
                        # Update root to new root
                        root = r_root
                        merged = True
        else:
            c = int(input[idx])
            idx += 1
            output.append(str(count[c]))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()