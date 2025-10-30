# YOUR CODE HERE
n, m = map(int, input().split())
stronger = []
for i in range(m):
    a, b = map(int, input().split())
    stronger.append((a, b))

adj = [[] for _ in range(n + 1)]
for a, b in stronger:
    adj[a].append(b)

def solve():
    possible_strongest = set(range(1, n + 1))
    for i in range(1, n + 1):
        is_strongest = True
        for j in range(1, n + 1):
            if i == j:
                continue
            is_stronger = False
            q = [i]
            visited = {i}
            while q:
                curr = q.pop(0)
                if curr == j:
                    is_stronger = True
                    break
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            if not is_stronger:
                is_strongest = False
                break
        if not is_strongest:
            possible_strongest.discard(i)

    if len(possible_strongest) == 1:
        print(possible_strongest.pop())
    else:
        print(-1)

solve()