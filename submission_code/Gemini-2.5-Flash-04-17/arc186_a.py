import math

def solve():
    N, Q = map(int, input().split())
    queries = [int(input()) for _ in range(Q)]

    possible_k_values = set()

    # The set of possible number of fixed elements K is the union of two sets:
    # 1. Values of rc + (N-r)(N-c) for 0 <= r, c <= N
    # 2. Values of r(N-c) + (N-r)c for 0 <= r, c <= N
    # Matrices corresponding to form 1 can be permuted into block structure with J in top-left and bottom-right.
    # Matrices corresponding to form 2 can be permuted into block structure with J in top-right and bottom-left.

    # Calculate values of type 1: rc + (N-r)(N-c)
    for r in range(N + 1):
        for c in range(N + 1):
            k1 = r * c + (N - r) * (N - c)
            possible_k_values.add(k1)

    # Calculate values of type 2: r(N-c) + (N-r)c
    # Note that r(N-c) + (N-r)c = Nr - rc + Nc - rc = Nr + Nc - 2rc
    # Also, N^2 - (rc + (N-r)(N-c)) = N^2 - rc - (N^2 - Nr - Nc + rc) = N^2 - rc - N^2 + Nr + Nc - rc = Nr + Nc - 2rc
    # So values of type 2 are simply N^2 - values of type 1.
    n_squared = N * N
    values_to_add = set()
    for v in possible_k_values:
        values_to_add.add(n_squared - v)
    possible_k_values.update(values_to_add)

    for k in queries:
        if k in possible_k_values:
            print("Yes")
        else:
            print("No")

solve()