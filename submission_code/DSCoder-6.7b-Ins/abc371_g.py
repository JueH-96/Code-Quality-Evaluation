def solve():
    N = int(input())
    P = list(map(int, input().split()))
    A = list(map(int, input().split()))

    # Create a list to store the final permutation
    final_permutation = [0] * N

    # For each element in A, find its corresponding position in P and place it in the final_permutation
    for i in range(N):
        final_permutation[P[i] - 1] = A[i]

    print(' '.join(map(str, final_permutation)))

solve()