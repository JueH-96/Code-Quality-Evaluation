# YOUR CODE HERE
n = int(input())
h = list(map(int, input().split()))

ans = -1
for i in range(1, n):
    if h[i] > h[0]:
        ans = i + 1
        break

print(ans)