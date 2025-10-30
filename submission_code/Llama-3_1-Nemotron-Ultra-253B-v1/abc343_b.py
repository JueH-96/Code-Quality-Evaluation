n = int(input())
adj = [list(map(int, input().split())) for _ in range(n)]

for i in range(n):
    connected = []
    for j in range(n):
        if adj[i][j] == 1:
            connected.append(j + 1)
    print(' '.join(map(str, sorted(connected))))