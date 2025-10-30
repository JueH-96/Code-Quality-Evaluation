# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    index += 2

    graph = defaultdict(list)
    for _ in range(M):
        A = int(input[index]) - 1
        B = int(input[index + 1]) - 1
        X = int(input[index + 2])
        Y = int(input[index + 3])
        graph[A].append((B, X, Y))
        graph[B].append((A, -X, -Y))
        index += 4

    visited = [False] * N
    coordinates = [(None, None)] * N

    def bfs(start):
        queue = deque([(start, 0, 0)])
        visited[start] = True
        coordinates[start] = (0, 0)
        while queue:
            current, x, y = queue.popleft()
            for neighbor, dx, dy in graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    nx, ny = x + dx, y + dy
                    coordinates[neighbor] = (nx, ny)
                    queue.append((neighbor, nx, ny))
                else:
                    if coordinates[neighbor] != (x + dx, y + dy):
                        return False
        return True

    for i in range(N):
        if not visited[i]:
            if not bfs(i):
                for j in range(N):
                    if not visited[j]:
                        coordinates[j] = ('undecidable', 'undecidable')

    for coord in coordinates:
        if coord == ('undecidable', 'undecidable'):
            print('undecidable')
        else:
            print(coord[0], coord[1])

if __name__ == "__main__":
    main()