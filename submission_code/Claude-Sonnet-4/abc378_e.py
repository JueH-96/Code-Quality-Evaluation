# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

total = 0

for l in range(N):
    current_sum = 0
    for r in range(l, N):
        current_sum += A[r]
        total += current_sum % M

print(total)