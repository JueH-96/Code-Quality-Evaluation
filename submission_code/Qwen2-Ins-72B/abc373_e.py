import heapq

def election_victory(N, M, K, A):
    A = [-a for a in A]
    heapq.heapify(A)
    remaining_votes = K - sum(A)
    A = [-a for a in A]
    A.sort()
    C = [0] * N
    for i in range(N):
        if i < M:
            C[i] = -1
        else:
            votes_needed = max(0, A[M-1] - A[i] + 1)
            if remaining_votes < votes_needed:
                C[i] = -1
            else:
                C[i] = votes_needed
                remaining_votes -= votes_needed
    return C

N, M, K = map(int, input().split())
A = list(map(int, input().split()))
C = election_victory(N, M, K, A)
print(*C)