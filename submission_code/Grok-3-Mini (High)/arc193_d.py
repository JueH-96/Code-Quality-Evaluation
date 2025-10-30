import sys
data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1
for test in range(T):
    N = int(data[index])
    index += 1
    A = data[index]
    index += 1
    B = data[index]
    index += 1
    # Compute for A
    min_P = N + 1
    max_P = 0
    M = 0
    for i in range(N):  # i is index in string, 0-based
        if A[i] == '1':
            M += 1
            pos = i + 1  # 1-based position
            if pos < min_P:
                min_P = pos
            if pos > max_P:
                max_P = pos
    # Compute for B
    min_S = N + 1
    max_S = 0
    Q = 0
    for i in range(N):
        if B[i] == '1':
            Q += 1
            pos = i + 1
            if pos < min_S:
                min_S = pos
            if pos > max_S:
                max_S = pos
    # Now check
    if M < Q or (max_S - min_S) > (max_P - min_P):
        print(-1)
    else:
        K = max(abs(min_P - min_S), abs(max_P - max_S))
        print(K)