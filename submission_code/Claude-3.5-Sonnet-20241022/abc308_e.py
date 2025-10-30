def mex(a, b, c):
    used = {a, b, c}
    i = 0
    while i in used:
        i += 1
    return i

N = int(input())
A = list(map(int, input().split()))
S = input()

ans = 0
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            if S[i] == 'M' and S[j] == 'E' and S[k] == 'X':
                ans += mex(A[i], A[j], A[k])

print(ans)