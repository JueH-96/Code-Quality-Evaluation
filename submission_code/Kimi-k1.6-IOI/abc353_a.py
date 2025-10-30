n = int(input())
h = list(map(int, input().split()))
first = h[0]
result = -1
for i in range(1, n):
    if h[i] > first:
        result = i + 1
        break
print(result)