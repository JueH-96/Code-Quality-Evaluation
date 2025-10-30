import sys

def count_black_squares(N, P, A, B, C, D):
    count = 0
    for i in range(A, C+1):
        for j in range(B, D+1):
            if P[i % N][j % N] == 'B':
                count += 1
    return count

N, Q = map(int, input().split())
P = []
for _ in range(N):
    P.append(input())

for _ in range(Q):
    A, B, C, D = map(int, input().split())
    print(count_black_squares(N, P, A, B, C, D))