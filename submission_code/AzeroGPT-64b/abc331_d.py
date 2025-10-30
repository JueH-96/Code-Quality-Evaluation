from sys import stdin

# Read input
N, Q = map(int, stdin.readline().split())
P = [stdin.readline() for _ in range(N)]
queries = [list(map(int, stdin.readline().split())) for _ in range(Q)]

# Preprocess the pattern
pattern_counts = []
for i in range(N):
    count = 0
    for j in range(N):
        if P[i][j] == 'B':
            count += 1
    pattern_counts.append(count)

# Solve each query
for A, B, C, D in queries:
    A, B, C, D = A - 1, B - 1, C - 1, D - 1
    ans = 0

    # Add full pattern repeats
    full_repeats = (C // N - A // N) * (D // N - B // N)
    ans += full_repeats * sum(pattern_counts)

    # Add partial rows at top
    if A // N != C // N:
        for i in range(A % N, N):
            ans += (D // N - B // N) * (P[i][B % N:D % N + 1].count('B'))

    # Add partial rows at bottom
    if A // N != C // N:
        for i in range(C % N + 1):
            ans += (D // N - B // N) * (P[i][B % N:D % N + 1].count('B'))

    # Add partial columns at left
    if B // N != D // N:
        for i in range(A // N, C // N + 1):
            ans += P[(i * N + A) % N][(B % N):N].count('B')

    # Add partial columns at right
    if B // N != D // N:
        for i in range(A // N, C // N + 1):
            ans += P[(i * N + A) % N][:D % N + 1].count('B')

    # Subtract overcounted corners
    overcounted = 0
    if (A // N != C // N) and (B // N != D // N):
        if A // N == C // N:
            i = A % N
        else:
            i = [A % N, C % N + 1][N - 1 <= C % N < N]
        if B // N == D // N:
            overcounted = P[i][B % N:D % N + 1].count('B')
        elif D % N < B % N:
            overcounted = P[i][B % N:D % N + 1].count('B')
        elif C % N < A % N:
            overcounted = sum([P[j][B % N:D % N + 1].count('B') for j in range(A % N, C % N + 1)])
    ans -= overcounted

    # Print result of each query
    print(ans)