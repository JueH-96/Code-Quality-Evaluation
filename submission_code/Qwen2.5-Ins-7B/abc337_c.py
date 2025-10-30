# YOUR CODE HERE
n = int(input())
arr = list(map(int, input().split()))
pos = [-1] * (n + 1)
ans = []

for i in range(n):
    if arr[i] == -1:
        ans.append(i + 1)
        continue
    pos[arr[i]] = i + 1

for i in range(1, n + 1):
    if pos[i] == -1:
        ans.append(i)

print(*ans)