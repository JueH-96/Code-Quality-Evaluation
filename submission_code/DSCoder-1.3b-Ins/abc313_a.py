# YOUR CODE HERE

N = int(input().strip())
P = list(map(int, input().strip().split()))

P.sort()

for i in range(N):
    if P[i] >= i + 1:
        print(P[i] - i)
        break