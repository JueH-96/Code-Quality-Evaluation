import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def read_int():
    return int(sys.stdin.readline())

def main():
    N, M = read_ints()
    p = [0] + [0] + read_ints()
    x = []
    y = []
    for _ in range(M):
        xi, yi = read_ints()
        x.append(xi)
        y.append(yi)

    # Create a tree structure
    tree = [[] for _ in range(N + 1)]
    for i in range(2, N + 1):
        tree[p[i]].append(i)

    # Perform DFS to find the depth of each node
    depth = [0] * (N + 1)
    stack = [(1, 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        for child in tree[node]:
            stack.append((child, d + 1))

    # Find the people covered by each insurance
    covered = set()
    for xi, yi in zip(x, y):
        stack = [(xi, 0)]
        while stack:
            node, d = stack.pop()
            if (node, d) not in covered:
                covered.add((node, d))
            if d < yi:
                for child in tree[node]:
                    stack.append((child, d + 1))

    # Count the number of people covered by at least one insurance
    count = len(set(node for node, _ in covered))

    print(count)

if __name__ == "__main__":
    main()