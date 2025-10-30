import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    K = int(data[index + 1])
    index += 2

    edges = []
    for _ in range(N - 1):
        A = int(data[index])
        B = int(data[index + 1])
        index += 2
        edges.append((A, B))

    V = []
    for _ in range(K):
        V.append(int(data[index]))
        index += 1

    from collections import defaultdict, deque

    # Build the tree
    tree = defaultdict(list)
    for A, B in edges:
        tree[A].append(B)
        tree[B].append(A)

    # BFS to find the minimum subtree
    queue = deque(V)
    visited = set(V)
    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    print(len(visited))

if __name__ == "__main__":
    solve()