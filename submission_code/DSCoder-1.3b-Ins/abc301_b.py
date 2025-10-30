# YOUR CODE HERE

N = int(input().split()[0])
A = list(map(int, input().split()))

while True:
    changed = False
    for i in range(N-1):
        if abs(A[i] - A[i+1]) == 1:
            mid = A[i] if A[i] < A[i+1] else A[i+1]
            A = A[:i+1] + [x+1 for x in range(mid, A[i+1])] + A[i+2:]
            N += len(range(mid, A[i+1])) - 1
            changed = True
            break
    if not changed:
        break

print(*A)