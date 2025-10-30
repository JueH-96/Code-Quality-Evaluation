n = int(input())
a = list(map(int, input().split()))
b = [0] * (n + 1)
for i in range(n):
    b[a[i]] = i + 1
index = 0
print(n - 1)
for i in range(1, n):
    if b[i] > b[i + 1]:
        print(b[i], b[i + 1])
        index = i + 1
        break
for j in range(n):
    if index == n - 1:
        break
    i = index
    while i > b[i]:
        print(b[i], b[i - 1])
        i -= 1
    if i < n - 1 and b[i + 1] < i + 1:
        print(i + 1, b[i + 1])
    index = i