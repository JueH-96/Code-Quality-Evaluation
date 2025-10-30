# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))
a = list(map(int, input().split()))

pos = [0] * (n + 1)
for i in range(1, n + 1):
    pos[p[i - 1]] = i

sorted_a = sorted(a)
ans = [-1] * n

for i in range(n):
    ans[pos[i + 1] - 1] = sorted_a[i]

print(*ans)