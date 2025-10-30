import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1

    edges = []
    for _ in range(M):
        u = int(input[idx]); idx +=1
        v = int(input[idx]); idx +=1
        w = int(input[idx]); idx +=1
        edges.append((w, u, v))
    edges.sort()

    A = list(map(int, input[idx:idx+K]))
    idx += K
    B = list(map(int, input[idx:idx+K]))
    idx += K

    count_A = [0] * (N + 1)
    count_B = [0] * (N + 1)
    for x in A:
        count_A[x] += 1
    for x in B:
        count_B[x] += 1

    parent = list(range(N + 1))
    size = [1] * (N + 1)
    a = [0] * (N + 1)
    b = [0] * (N + 1)
    for i in range(N + 1):
        a[i] = count_A[i]
        b[i] = count_B[i]

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    total = 0

    for w, u, v in edges:
        x_root = find(u)
        y_root = find(v)
        if x_root == y_root:
            continue

        a_x = a[x_root]
        b_x = b[x_root]
        a_y = a[y_root]
        b_y = b[y_root]

        pairs_before = min(a_x, b_x) + min(a_y, b_y)
        merged_a = a_x + a_y
        merged_b = b_x + b_y
        pairs_after = min(merged_a, merged_b)
        delta = pairs_after - pairs_before

        if delta > 0:
            total += delta * w

        if size[x_root] < size[y_root]:
            x_root, y_root = y_root, x_root
            a_x, a_y = a_y, a_x
            b_x, b_y = b_y, b_x

        parent[y_root] = x_root
        size[x_root] += size[y_root]
        a[x_root] = merged_a
        b[x_root] = merged_b

    print(total)

if __name__ == '__main__':
    main()