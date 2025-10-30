# YOUR CODE HERE
N = int(input())
graph = [list(map(int, input().split())) for _ in range(N)]

for i in range(N):
    connected = [j+1 for j in range(N) if graph[i][j] == 1]
    print(*connected)