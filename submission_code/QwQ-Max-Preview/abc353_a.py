n = int(input())
h = list(map(int, input().split()))
target = h[0]
result = -1
for i in range(1, n):
    if h[i] > target:
        result = i + 1
        break
print(result)