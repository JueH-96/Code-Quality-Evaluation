# YOUR CODE HERE
def find_cycle(N, A):
    visited = [False] * (N + 1)
    stack = [False] * (N + 1)
    parent = [-1] * (N + 1)
    cycle = []

    def dfs(v):
        visited[v] = True
        stack[v] = True

        next_v = A[v-1]
        if stack[next_v]:
            cycle.append(v)
            current = v
            while current != next_v:
                current = parent[current]
                cycle.append(current)
            cycle.reverse()
            return True
        elif not visited[next_v]:
            parent[next_v] = v
            if dfs(next_v):
                return True

        stack[v] = False
        return False

    for i in range(1, N + 1):
        if not visited[i]:
            if dfs(i):
                break

    print(len(cycle))
    print(*cycle)

N = int(input())
A = list(map(int, input().split()))
find_cycle(N, A)