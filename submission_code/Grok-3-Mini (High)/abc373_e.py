import sys
data = sys.stdin.read().split()
N, M, K = int(data[0]), int(data[1]), int(data[2])
A = list(map(int, data[3:3+N]))
sum_A = sum(A)
R = K - sum_A

if N - 1 < M:
    C = [0] * N
else:
    sorted_indices = sorted(range(N), key=lambda i: A[i], reverse=True)
    sum_top_M = sum(A[sorted_indices[k]] for k in range(M))
    val_M_plus_1 = A[sorted_indices[M]]
    rank = [0] * N
    for k in range(N):
        rank[sorted_indices[k]] = k
    C = []
    for i in range(N):
        P_i = rank[i]
        if P_i >= M:
            S_i = sum_top_M
        else:
            S_i = sum_top_M - A[i] + val_M_plus_1
        D_prime = S_i + R - M * A[i] - M
        X_min_req = (D_prime // (M + 1)) + 1
        if X_min_req <= R:
            C_i_val = max(0, X_min_req)
        else:
            C_i_val = -1
        C.append(C_i_val)

print(' '.join(map(str, C)))