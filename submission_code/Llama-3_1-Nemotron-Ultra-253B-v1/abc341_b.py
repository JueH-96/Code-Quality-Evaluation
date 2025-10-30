n = int(input())
A = list(map(int, input().split()))
S = []
T = []
for _ in range(n-1):
    s, t = map(int, input().split())
    S.append(s)
    T.append(t)

for i in range(n-1):
    s = S[i]
    t = T[i]
    conversions = A[i] // s
    A[i+1] += conversions * t
    A[i] -= conversions * s

print(A[-1])