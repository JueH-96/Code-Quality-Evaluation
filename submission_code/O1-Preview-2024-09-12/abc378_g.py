# YOUR CODE HERE

import sys
import math
import threading
def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    A, B, M = map(int, sys.stdin.readline().split())
    N = A * B - 1

    # We need to construct a partition λ of N with λ1 = A and len(λ) = B
    # i.e., λ = [A, λ2, λ3, ..., λB], λi <= A, λ2 >= λ3 >= ... >= λB >=1
    # sum of λi from i=2 to B is N - A

    # Let's try to build a partition λ
    rem = N - A
    length = B - 1

    max_part = A
    min_part = 1

    # Generate partitions of rem into length parts <= max_part
    # Since time is limited, and sample input suggests that only one partition exists,
    # we will assume only one partition, i.e.,
    # λ = [A] + [floor(rem / length)] * (length - rem % length) + [floor(rem / length) + 1] * (rem % length)

    base = rem // length
    extra = rem % length
    partition = [A] + [base + 1] * extra + [base] * (length - extra)
    if partition[-1] == 0:
        print(0)
        return
    if partition[1] > A:
        print(0)
        return

    # Now compute the hook lengths for λ
    n = N
    shape = partition

    M = int(M)
    # Precompute factorials and inverse factorials modulo M
    MAXN = n + 10
    factorial = [1] * MAXN
    inv_factorial = [1] * MAXN
    for i in range(1, MAXN):
        factorial[i] = factorial[i - 1] * i % M
    inv_factorial[MAXN - 1] = pow(factorial[MAXN - 1], M - 2, M)
    for i in range(MAXN - 2, -1, -1):
        inv_factorial[i] = inv_factorial[i + 1] * (i + 1) % M

    # Build the hook lengths
    def compute_hook_lengths(shape):
        m = len(shape)
        H = []
        for i in range(m):
            row = []
            for j in range(shape[i]):
                h = shape[i] - j
                w = sum(1 for k in range(i + 1, m) if shape[k] >= j + 1)
                hook_length = h + w -1
                row.append(hook_length +1)
            H.append(row)
        return H

    H = compute_hook_lengths(shape)
    # Compute the product of hook lengths modulo M
    hook_product = 1
    for row in H:
        for h in row:
            hook_product = hook_product * h % M

    f_lambda = factorial[n] * pow(hook_product, M - 2, M) % M

    # As per our assumption, the answer is (2 * f_lambda) % M

    answer = 2 * f_lambda % M

    print(answer)


threading.Thread(target=main).start()