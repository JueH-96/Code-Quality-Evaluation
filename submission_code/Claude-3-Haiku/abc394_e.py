from collections import deque

def shortest_palindrome_path(N, graph):
    # Initialize the answer matrix
    ans = [[-1] * N for _ in range(N)]

    # Perform BFS for each pair of vertices
    for i in range(N):
        for j in range(N):
            queue = deque([(i, [i])])
            while queue:
                u, path = queue.popleft()
                if u == j:
                    if ''.join(graph[p][q] for p, q in zip(path, path[::-1])) == ''.join(graph[p][q] for p, q in zip(path, path)):
                        ans[i][j] = len(path) - 1
                        break
                for v in range(N):
                    if graph[u][v] != '-':
                        queue.append((v, path + [v]))

    return ans

# Read the input
N = int(input())
graph = [list(input()) for _ in range(N)]

# Solve the problem and print the output
result = shortest_palindrome_path(N, graph)
for row in result:
    print(' '.join(map(str, row)))