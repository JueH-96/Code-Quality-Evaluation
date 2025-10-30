n = int(input())
a = list(map(int, input().split()))
b = [0] * n
for i, x in enumerate(a):
    b[x - 1] = i
ans = []
for i in range(n - 1):
    if b[i] > b[i + 1]:
        ans.append((b[i + 1] + 1, b[i] + 1))
        b[i], b[i + 1] = b[i + 1], b[i]
print(len(ans))
for x, y in ans:
    print(x, y)