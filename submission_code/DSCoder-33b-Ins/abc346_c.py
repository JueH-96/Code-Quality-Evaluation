# YOUR CODE HERE

N, K = map(int, input().split())
A = list(map(int, input().split()))

A_set = set(A)
total = 0

for i in range(1, K+1):
    if i not in A_set:
        total += i

print(total)