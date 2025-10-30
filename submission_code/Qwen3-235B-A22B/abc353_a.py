n = int(input())
h = list(map(int, input().split()))
first = h[0]
for i in range(1, n):
    if h[i] > first:
        print(i + 1)
        break
else:
    print(-1)