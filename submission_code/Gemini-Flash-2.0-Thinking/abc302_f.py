from collections import deque

def solve():
    n, m = map(int, input().split())
    sets_data = []
    for _ in range(n):
        line = list(map(int, input().split()))
        sets_data.append(set(line[1:]))

    for s in sets_data:
        if 1 in s and m in s:
            print(0)
            return

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if any(val in sets_data[j] for val in sets_data[i]):
                adj[i].append(j)
                adj[j].append(i)

    starts = [i for i, s in enumerate(sets_data) if 1 in s]
    ends = [i for i, s in enumerate(sets_data) if m in s]

    if not starts or not ends:
        print(-1)
        return

    queue = deque([(start, 0) for start in starts])
    visited = set(starts)

    while queue:
        current_index, merges = queue.popleft()

        if current_index in ends:
            print(merges)
            return

        for neighbor_index in adj[current_index]:
            if neighbor_index not in visited:
                visited.add(neighbor_index)
                queue.append((neighbor_index, merges + 1))

    print(-1)

solve()