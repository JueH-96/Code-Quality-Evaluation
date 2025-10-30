def solve():
    N, M = map(int, input().split())
    relations = [tuple(map(int, input().split())) for _ in range(M)]

    strongest = [0] * (N + 1)
    for A, B in relations:
        strongest[B] = max(strongest[B], A)

    for i in range(N, 0, -1):
        if strongest[i] < i:
            return -1

    return i

print(solve())