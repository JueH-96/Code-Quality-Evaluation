# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))

max_other = 0
for i in range(1, n):
  max_other = max(max_other, p[i])

ans = max(0, max_other - p[0] + 1)
print(ans)