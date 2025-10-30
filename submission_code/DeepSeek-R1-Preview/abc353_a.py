n = int(input())
h = list(map(int, input().split()))
target = h[0]
for i in range(1, n):
    if h[i] > target:
        print(i + 1)
        exit()
print(-1)