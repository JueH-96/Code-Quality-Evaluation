import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    parent = list(range(N + 2))  # 1-based indexing
    left = list(range(N + 2))
    right = list(range(N + 2))
    color = list(range(N + 2))
    count = [0] * (N + 2)
    for x in range(1, N+1):
        count[x] = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _ in range(Q):
        parts = sys.stdin.readline().split()
        if not parts:
            continue
        if parts[0] == '1':
            x = int(parts[1])
            c = int(parts[2])
            root = find(x)
            old_c = color[root]
            size = right[root] - left[root] + 1
            count[old_c] -= size
            color[root] = c
            count[c] += size
            # Merge left
            current_l = left[root]
            while current_l > 1:
                left_neighbor = current_l - 1
                left_root = find(left_neighbor)
                if color[left_root] == c and right[left_root] == left_neighbor:
                    parent[left_root] = root
                    left[root] = left[left_root]
                    current_l = left[root]
                else:
                    break
            # Merge right
            current_r = right[root]
            while current_r < N:
                right_neighbor = current_r + 1
                right_root = find(right_neighbor)
                if color[right_root] == c and left[right_root] == right_neighbor:
                    parent[right_root] = root
                    right[root] = right[right_root]
                    current_r = right[root]
                else:
                    break
        else:
            c = int(parts[1])
            print(count[c])

if __name__ == '__main__':
    main()