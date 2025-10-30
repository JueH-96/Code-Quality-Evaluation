N, M = map(int, input().split())
A = list(map(int, input().split()))

missing = []
for i in range(1, N + 1):
    if i not in A:
        missing.append(i)

print(len(missing))
if len(missing) > 0:
    print(*missing)