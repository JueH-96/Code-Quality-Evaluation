# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

while True:
    found = False
    for i in range(N - 1):
        if abs(A[i] - A[i + 1]) != 1:
            found = True
            if A[i] < A[i + 1]:
                A = A[:i + 1] + list(range(A[i] + 1, A[i + 1])) + A[i + 1:]
            else:
                A = A[:i + 1] + list(range(A[i] - 1, A[i + 1], -1)) + A[i + 1:]
            N = len(A)
            break
    if not found:
        break

print(' '.join(map(str, A)))