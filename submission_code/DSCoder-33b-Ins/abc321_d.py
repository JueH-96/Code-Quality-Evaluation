# YOUR CODE HERE

N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

total_price = 0

for i in range(N):
    for j in range(M):
        total_price += min(A[i] + B[j], P)

print(total_price)