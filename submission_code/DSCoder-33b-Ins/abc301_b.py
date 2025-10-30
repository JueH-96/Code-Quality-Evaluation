# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

i = 0
while i < len(A) - 1:
    if abs(A[i] - A[i+1]) != 1:
        if A[i] < A[i+1]:
            for j in range(A[i]+1, A[i+1]):
                A.insert(i+1, j)
                i += 1
        else:
            for j in range(A[i]-1, A[i+1], -1):
                A.insert(i+1, j)
                i += 1
    else:
        i += 1

print(' '.join(map(str, A)))