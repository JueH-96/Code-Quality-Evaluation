import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

A = [0] * (N + 1)
operations = [0] * (N + 1)

current_operation = 0

while True:
    A[0] += 1
    current_operation += 1

    for i in range(1, N + 1):
        if A[i-1] > A[i] and A[i-1] > H[i-1]:
            A[i-1] -= 1
            A[i] += 1
            if A[i] == 1:
                operations[i] = current_operation
        if i == N and all(x > 0 for x in operations[1:]):
            break

    if all(x > 0 for x in operations[1:]):
        break

print(" ".join(map(str, operations[1:])))