# YOUR CODE HERE
n = int(input())
intervals = [list(map(int, input().split())) for _ in range(n)]
l, r = zip(*intervals)
left, right = sorted(l), sorted(r)
ans = 0
for i in range(n):
    j = bisect_right(right, left[i])
    ans += i - j
print(ans)