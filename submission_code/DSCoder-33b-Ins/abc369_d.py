# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

A.sort(reverse=True)

total_exp = 0
for i in range(N):
    if i % 2 == 0:
        total_exp += A[i]

print(total_exp)