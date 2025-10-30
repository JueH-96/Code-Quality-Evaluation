from math import gcd

def common_divisor(a, b):
    return a // gcd(a, b)

T = int(input())

for _ in range(T):
    N = int(input())
    A = []
    B = []
    C = []

    for _ in range(N):
        a, b, c = map(int, input().split())
        A.append(a)
        B.append(b)
        C.append(c)
    
    A1 = [None for _ in range(N)]
    P = []
    S = []
    
    for i, (a, b, c) in enumerate(zip(A, B, C)):
        # B[i]y < C[i]-A[i]x
        # B[i]y < C[i](1 - a)
        k1 = c - a
        A1[i] = min(b, k1)
        P.append(a)
        S.append(b)
    
    X = [0] * N
    p = [0] * N
    p[0] = P[0]
    
    # fastest way to compute prefix gcd from a0, ..., ai-1
    for i in range(1, N):
        p[i] = gcd(p[i-1], P[i])
    
    X[N-1] = int((S[N-1] // A1[N-1]) * A1[N-1])
    inva_N1 = common_divisor(A1[N-1], S[N-1])
    S_N1inva_N1_C_N1 = min(S[N-1], C[N-1])
    
    if A1[N-1]:
        X[N-1] = min(X[N-1], S_N1inva_N1_C_N1 - (S[N-1] % A1[N-1]))
    
    answer = (X[N-1] // inva_N1) + 1
    
    for i in range(N-2, -1, -1):
        inva_i = common_divisor(A1[i], S[i])
        X[i] = int(min(X[i+1] + (P[i] // p[i+1]) * S[i], (S[i] // A1[i]) * A1[i]))
        X[i] = min(X[i], S[i] - (S[i] % A1[i]), C[i] - (P[i] * (X[i+1] // S[i])))
        answer += (X[i] // inva_i) + 1
    
    print(answer)