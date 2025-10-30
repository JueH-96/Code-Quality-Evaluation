# YOUR CODE HERE
N,*rest = map(int, open(0).read().split())
A = rest[:N]
while True:
    changed = False
    for i in range(len(A)-1):
        if abs(A[i] - A[i+1]) != 1:
            if A[i] < A[i+1]:
                insertion = list(range(A[i]+1, A[i+1]))
            else:
                insertion = list(range(A[i]-1, A[i+1], -1))
            A = A[:i+1] + insertion + A[i+1:]
            changed = True
            break
    if not changed:
        break
print(' '.join(map(str,A)))