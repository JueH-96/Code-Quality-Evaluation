# YOUR CODE HERE
N, X = map(int, input().split())
A = list(map(int, input().split()))

A.sort()

total = sum(A[1:-1])

for i in range(101):
    if total + i >= X:
        print(i)
        break
else:
    print(-1)