N = int(input())
A = list(map(int, input().split()))
ST = [list(map(int, input().split())) for _ in range(N-1)]

for i in range(N-1)[::-1]:
    s, t = ST[i]
    if A[i] >= s:
        A[i+1] += A[i] // s * t

print(A[-1])