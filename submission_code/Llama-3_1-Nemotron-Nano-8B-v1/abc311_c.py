n = int(input())
A = list(map(int, input().split()))

visited = [False] * (n + 1)

for i in range(1, n + 1):
    if not visited[i]:
        path = []
        pos = {}
        current = i
        while True:
            if visited[current]:
                if current in pos:
                    idx = pos[current]
                    cycle = path[idx:]
                    print(len(cycle))
                    print(' '.join(map(str, cycle)))
                    exit()
                else:
                    break
            else:
                visited[current] = True
                pos[current] = len(path)
                path.append(current)
                current = A[current - 1]