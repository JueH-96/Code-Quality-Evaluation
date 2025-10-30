N, M = map(int, input().split())
A = [0] * M
B = [0] * M
for i in range(M):
    A[i], B[i] = input().split()
    A[i] = int(A[i])
    
first_sons = [True] * N
for i in range(M):
    if first_sons[A[i] - 1] and B[i] == 'M':
        print('Yes')
        first_sons[A[i] - 1] = False
    else:
        print('No')