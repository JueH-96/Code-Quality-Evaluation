# YOUR CODE HERE
n, q = map(int, input().split())
r = list(map(int, input().split()))
r.sort()
s = [0] * (n + 1)
for i in range(n):
    s[i + 1] = s[i] + r[i]
for _ in range(q):
    x = int(input())
    left = 0
    right = n
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        if s[mid] <= x:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    print(ans)