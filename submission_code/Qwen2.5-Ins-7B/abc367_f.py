# N, Q = map(int, input().split())
# A = list(map(int, input().split()))
# B = list(map(int, input().split()))
# queries = [tuple(map(int, input().split())) for _ in range(Q)]

def solve(N, Q, A, B, queries):
    for l, r, L, R in queries:
        subseq_A = sorted(A[l-1:r])
        subseq_B = sorted(B[L-1:R])
        if subseq_A == subseq_B:
            print('Yes')
        else:
            print('No')

# solve(N, Q, A, B, queries)

N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
queries = [tuple(map(int, input().split())) for _ in range(Q)]
solve(N, Q, A, B, queries)