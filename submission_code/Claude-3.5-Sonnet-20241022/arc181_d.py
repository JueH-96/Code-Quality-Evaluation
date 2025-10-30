def count_inversions(P):
    n = len(P)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if P[i] > P[j]:
                count += 1
    return count

def operation_k(P, k):
    P = list(P)
    for i in range(k-1):
        if P[i] > P[i+1]:
            P[i], P[i+1] = P[i+1], P[i]
    return P

N = int(input())
P = list(map(int, input().split()))
M = int(input())
A = list(map(int, input().split()))

curr_P = P[:]
for i in range(M):
    k = A[i]
    curr_P = operation_k(curr_P, k)
    print(count_inversions(curr_P))