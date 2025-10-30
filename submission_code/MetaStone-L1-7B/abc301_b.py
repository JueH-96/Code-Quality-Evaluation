n = int(input())
A = list(map(int, input().split()))

while True:
    found = False
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            if A[i] < A[i+1]:
                inserted = list(range(A[i] + 1, A[i+1]))
            else:
                inserted = list(range(A[i] - 1, A[i+1] - 1, -1))
            A = A[:i+1] + inserted + A[i+2:]
            found = True
            break
    if not found:
        break

print(' '.join(map(str, A)))