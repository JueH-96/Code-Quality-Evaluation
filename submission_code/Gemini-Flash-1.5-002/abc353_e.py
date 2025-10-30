# YOUR CODE HERE
def f(x, y):
    i = 0
    while i < len(x) and i < len(y) and x[i] == y[i]:
        i += 1
    return i

N = int(input())
S = input().split()
ans = 0
for i in range(N - 1):
    for j in range(i + 1, N):
        ans += f(S[i], S[j])
print(ans)