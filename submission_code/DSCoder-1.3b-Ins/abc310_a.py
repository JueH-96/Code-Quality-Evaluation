# YOUR CODE HERE

N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

D.sort()

total = 0

for i in range(N):
    if i % 2 == 0:
        total += D[i]
    else:
        total += min(D[i], Q)

print(total)