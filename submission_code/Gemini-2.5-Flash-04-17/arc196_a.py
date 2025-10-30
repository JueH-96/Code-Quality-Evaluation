import sys

# For competitive programming, often faster I/O is needed
# input = sys.stdin.readline

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    O = []
    E = []
    for i in range(N):
        # Original index is i+1.
        # Odd original indices correspond to 0-indexed i such that i+1 is odd, i.e., i is even.
        # Even original indices correspond to 0-indexed i such that i+1 is even, i.e., i is odd.
        if i % 2 == 0: # Original index 1, 3, 5, ...
            O.append(A[i])
        else: # Original index 2, 4, 6, ...
            E.append(A[i])

    O.sort()
    E.sort()

    if N % 2 == 0:
        # N is even. |O| = N/2, |E| = N/2. K = N/2.
        # We pair the j-th smallest element in O with the (K-1-j)-th smallest element in E
        # to maximize the sum of absolute differences between the two sets of size K.
        K = N // 2
        current_score = 0
        for j in range(K):
            current_score += abs(O[j] - E[K - 1 - j])
        print(current_score)
    else:
        # N is odd. |O| = (N+1)/2, |E| = (N-1)/2.
        # One element from O remains. All E are paired with (N-1)/2 elements from O.
        # Let K = (N-1)/2. |O| = K+1, |E| = K.
        
        K = (N - 1) // 2
        
        # We iterate through each element in O to be the one that remains.
        # If O[i] remains, the paired sets are O' = O \ {O[i]} (size K) and E (size K).
        # To maximize the sum of absolute differences between two sets of size K,
        # we pair the j-th smallest element of O' with the (K-1-j)-th smallest of E.
        
        # Let E_rev be E sorted in descending order. E_rev[j] = E[K-1-j].
        
        # Precompute values for a_j = |O[j] - E[K-1-j]| and b_j = |O[j+1] - E[K-1-j]|
        # These are differences used in sums depending on which element is removed from O.
        # j from 0 to K-1
        a = [0] * K # a[j] = |O[j] - E[K-1-j]|
        b = [0] * K # b[j] = |O[j+1] - E[K-1-j]|
        for j in range(K):
            a[j] = abs(O[j] - E[K - 1 - j])
            b[j] = abs(O[j+1] - E[K - 1 - j])

        # Precompute prefix sums of a and suffix sums of b
        # PS_a[k] = sum(a[0]...a[k-1])
        PS_a = [0] * (K + 1)
        for k in range(K):
            PS_a[k+1] = PS_a[k] + a[k]

        # SS_b[k] = sum(b[k]...b[K-1])
        SS_b = [0] * (K + 1)
        for k in range(K - 1, -1, -1):
            SS_b[k] = SS_b[k+1] + b[k]

        max_score = 0

        # Iterate through which element O[i] (0-indexed in sorted O) is removed
        # i can range from 0 to K. |O| = K+1.
        for i in range(K + 1):
            current_score = 0
            if i == 0:
                # O[0] is removed. O' = {O[1], ..., O[K]}. j-th element of O' is O[j+1].
                # Score = sum |O[j+1] - E[K-1-j]| for j=0..K-1. This is sum of b[j] for j=0..K-1 = SS_b[0].
                current_score = SS_b[0]
            elif i == K:
                # O[K] is removed. O' = {O[0], ..., O[K-1]}. j-th element of O' is O[j].
                # Score = sum |O[j] - E[K-1-j]| for j=0..K-1. This is sum of a[j] for j=0..K-1 = PS_a[K].
                current_score = PS_a[K]
            else: # 1 <= i <= K-1
                # O[i] is removed. O' = {O[0]...O[i-1], O[i+1]...O[K]}.
                # j-th element of O' is O[j] for j < i, O[j+1] for j >= i.
                # Score = sum |O[j] - E[K-1-j]| for j=0..i-1 + sum |O[j+1] - E[K-1-j]| for j=i..K-1
                # This is sum(a[0]...a[i-1]) + sum(b[i]...b[K-1]) = PS_a[i] + SS_b[i].
                current_score = PS_a[i] + SS_b[i]

            max_score = max(max_score, current_score)
        
        print(max_score)

solve()