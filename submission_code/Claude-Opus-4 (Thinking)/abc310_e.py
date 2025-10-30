# YOUR CODE HERE
N = int(input())
S = input().strip()
A = [int(c) for c in S]

total = 0
for i in range(N):
    current = A[i]
    total += current
    for j in range(i + 1, N):
        if current == 0:
            current = 1
        else:
            current = 1 - A[j]
        total += current

print(total)