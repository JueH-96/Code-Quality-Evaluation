N = int(input())
S = input().split()

def f(x, y):
    i = 0
    min_len = min(len(x), len(y))
    while i < min_len and x[i] == y[i]:
        i += 1
    return i

ans = 0
for i in range(N-1):
    for j in range(i+1, N):
        ans += f(S[i], S[j])
        
print(ans)