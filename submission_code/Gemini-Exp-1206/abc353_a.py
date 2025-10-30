n = int(input())
h = list(map(int, input().split()))

first_height = h[0]
result = -1

for i in range(1, n):
    if h[i] > first_height:
        result = i + 1
        break

print(result)