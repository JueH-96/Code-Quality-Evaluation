n = int(input())
for _ in range(n):
    row = list(map(int, input().split()))
    connected = []
    for j in range(n):
        if row[j] == 1:
            connected.append(j + 1)
    connected.sort()
    print(' '.join(map(str, connected)))