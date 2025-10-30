# YOUR CODE HERE
import sys

import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    K = int(sys.stdin.readline())
    queries = [tuple(map(int, sys.stdin.readline().split())) + (i,) for i in range(K)]

    A.sort()
    B.sort()

    cumA = [0] * (N + 1)
    cumA_pos = [0] * (N + 1)
    for i in range(N):
        cumA[i + 1] = cumA[i] + A[i]
        cumA_pos[i + 1] = cumA_pos[i] + A[i] * (i + 1)

    cumB = [0] * (N + 1)
    cumB_pos = [0] * (N + 1)
    for i in range(N):
        cumB[i + 1] = cumB[i] + B[i]
        cumB_pos[i + 1] = cumB_pos[i] + B[i] * (i + 1)

    # Precompute total sums of absolute differences within A and B
    SA = [0] * (N + 1)
    for i in range(1, N + 1):
        SA[i] = SA[i - 1] + A[i - 1] * (2 * i - N - 1)

    SB = [0] * (N + 1)
    for i in range(1, N + 1):
        SB[i] = SB[i - 1] + B[i - 1] * (2 * i - N - 1)

    # Since we cannot precompute cross sums, process each query individually
    # Given the constraints, we need to process per query in O(N) time
    # Since K ≤ 1e4 and N ≤ 1e5, total operations are acceptable (~1e9)

    # Alternative approach: Process all queries at once using coordinates compression

    # Prepare unique X_k and Y_k
    all_X = set()
    all_Y = set()
    for X, Y, _ in queries:
        all_X.add(X)
        all_Y.add(Y)
    all_X = sorted(all_X)
    all_Y = sorted(all_Y)

    prefix_As = {}
    for X in all_X:
        prefix_As[X] = A[:X]
    prefix_Bs = {}
    for Y in all_Y:
        prefix_Bs[Y] = B[:Y]

    results = [0] * K
    for idx, (X, Y, k) in enumerate(queries):
        s = 0
        i = 0
        j = 0
        numA = 0
        numB = 0
        sumA = 0
        sumB = 0
        total = 0

        len_combined = X + Y
        combined = [(a, 'A') for a in A[:X]] + [(b, 'B') for b in B[:Y]]
        combined.sort()

        numA = numB = 0
        sumA = sumB = 0
        total = 0
        for val, origin in combined:
            if origin == 'A':
                total += val * numB - sumB
                numA += 1
                sumA += val
            else:
                total += val * numA - sumA
                numB += 1
                sumB += val
        results[k] = total

    for res in results:
        print(res)

threading.Thread(target=main).start()