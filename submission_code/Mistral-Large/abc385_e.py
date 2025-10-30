import sys
from collections import deque, defaultdict

def bfs(start, tree):
    queue = deque([start])
    visited = set([start])
    levels = defaultdict(list)
    level = 0

    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.popleft()
            levels[level].append(node)
            for neighbor in tree[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        level += 1

    return levels

def is_snowflake_tree(levels):
    if len(levels) != 3:
        return False
    if len(levels[0]) != 1 or len(levels[1]) < 1 or len(levels[2]) < 1:
        return False
    x = len(levels[1])
    y = len(levels[2]) // x
    return all(len(levels[2][i*y:(i+1)*y]) == y for i in range(x))

def min_deletions_to_snowflake(N, edges):
    if N <= 2:
        return 0

    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)

    min_deletions = float('inf')

    for start in range(1, N + 1):
        levels = bfs(start, tree)
        if is_snowflake_tree(levels):
            min_deletions = min(min_deletions, N - sum(len(level) for level in levels.values()))

    return min_deletions

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    edges = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N-1)]

    result = min_deletions_to_snowflake(N, edges)
    print(result)

if __name__ == "__main__":
    main()