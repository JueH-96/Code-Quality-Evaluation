# YOUR CODE HERE

N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

price = 0
for i in range(N):
    if C[i] in D:
        price += P[D.index(C[i])]
    else:
        price += P[0]

print(price)