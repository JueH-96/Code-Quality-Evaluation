# YOUR CODE HERE
n, p, q = map(int, input().split())
d = list(map(int, input().split()))

ans = p
for di in d:
    ans = min(ans, q + di)

print(ans)