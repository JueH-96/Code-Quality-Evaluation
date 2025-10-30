from collections import deque, defaultdict

def solve():
    N = int(input())
    prerequisites = defaultdict(list)
    indegrees = [0] * (N+1)
    for i in range(1, N+1):
        C_i, *P_i = map(int, input().split())
        for j in P_i:
            prerequisites[j].append(i)
            indegrees[i] += 1
    queue = deque([i for i in range(1, N+1) if indegrees[i] == 0])
    while queue:
        book = queue.popleft()
        for next_book in prerequisites[book]:
            indegrees[next_book] -= 1
            if indegrees[next_book] == 0:
                queue.append(next_book)
    print(*[i for i in range(2, N+1) if indegrees[i] > 0])

solve()