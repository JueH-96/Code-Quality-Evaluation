n = int(input())
A = list(map(int, input().split()))
S = []
T = []

for _ in range(n-1):
    s, t = map(int, input().split())
    S.append(s)
    T.append(t)

carry = 0

for i in range(n-1):
    total = A[i] + carry
    k = total // S[i]
    A[i+1] += k * T[i]
    carry = total % S[i]

print(A[-1])