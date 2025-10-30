# YOUR CODE HERE
n, x = map(int, input().split())
a = list(map(int, input().split()))

ans = -1
for i in range(101):
    b = a + [i]
    b.sort()
    s = sum(b[1:-1])
    if s >= x:
        ans = i
        break

print(ans)