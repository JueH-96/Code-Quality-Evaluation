N, M, L = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = [list(map(int, input().split())) for _ in range(L)]

A.sort(reverse=True)
B.sort(reverse=True)

max_price = A[0] + B[0]

for c in C:
    max_price = max(max_price, A[c[0]-1] + B[c[1]-1])

print(max_price)