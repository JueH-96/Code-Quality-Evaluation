# YOUR CODE HERE
def min_operations_to_match_sequences(N, M, A, B):
    def distance(a, b, M):
        d1 = (b - a) % M
        d2 = (a - b) % M
        return min(d1, d2)
    
    def is_good_sequence(seq):
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                return False
        return True
    
    def can_transform(A, B, M):
        for i in range(N):
            if (A[i] - B[i]) % M == 1 and (A[i-1] == B[i-1] or A[i+1] == B[i+1]):
                return False
            if (B[i] - A[i]) % M == 1 and (A[i-1] == B[i-1] or A[i+1] == B[i+1]):
                return False
        return True
    
    if A == B:
        return 0
    
    if not can_transform(A, B, M):
        return -1
    
    operations = 0
    for i in range(N):
        operations += distance(A[i], B[i], M)
    
    return operations

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:2*N+2]))

print(min_operations_to_match_sequences(N, M, A, B))