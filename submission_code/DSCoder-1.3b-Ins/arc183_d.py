import sys

def solve():
    N = int(sys.stdin.readline().strip())
    edges = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().strip().split())
        edges.append((a, b))
    edges.sort()

    visited = [False] * (N + 1)
    match = [-1] * (N + 1)
    score = 0

    for a, b in edges:
        if visited[a] or visited[b]:
            continue
        visited[a] = visited[b] = True
        match[a] = b
        match[b] = a
        score += b - a

    print(score)
    for a, b in edges:
        if a == match[a]:
            print(a, b)

if __name__ == "__main__":
    solve()