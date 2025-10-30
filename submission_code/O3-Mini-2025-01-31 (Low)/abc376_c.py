def main():
    import sys
    input = sys.stdin.readline

    # Read input
    N = int(input().strip())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Sort the toy sizes and existing box sizes.
    A.sort()
    B.sort()
    
    # We want to insert a new box of size x into the list B,
    # forming a sorted list of N boxes, call it D.
    # We wish to assign each toy (sorted as A) to a unique box D[i] with D[i] >= A[i].
    # Since x is our choice, we may insert it at any position k (0-indexed)
    # among the sorted boxes. There are N possible positions (k = 0 to N-1, with k = N meaning new box becomes largest).
    # In detail:
    #   If the new box is inserted at position k (0 <= k < N):
    #     - For indices i in [0, k-1], D[i] = B[i] and we require  B[i] >= A[i].
    #     - For index i == k, D[k] = x and we require x >= A[k].
    #       Also, because D remains sorted, if k > 0 we must have D[k-1] = B[k-1] <= x.
    #     - For indices i in [k+1, N-1], D[i] = B[i-1] and we require B[i-1] >= A[i].
    #   If the new box is inserted as the largest (position k == N):
    #     - Then D[i] = B[i] for i in [0, N-1] and D[N] = x.
    #     - We require for i in [0, N-2]: B[i] >= A[i]
    #       and x, which will be D[N-1], satisfies x >= A[N-1].
    # We wish to choose the minimal x for which there is any valid insertion.
    
    # Set ans to a large number and update whenever we get a valid candidate x.
    ans = 10**18  # sufficiently large given constraints
    
    # Case 1: Insertion at position k = 0.
    # Then new box becomes the smallest: x becomes D[0] and needs to be at least A[0].
    # For i in 1..N-1, D[i] = B[i-1] so the requirement is B[i-1] >= A[i].
    ok = True
    for i in range(1, N):
        if B[i-1] < A[i]:
            ok = False
            break
    if ok:
        candidate = A[0]  # minimal x must be at least A[0]
        ans = min(ans, candidate)
    
    # Case 2: Insertion at positions 1 <= k < N.
    for k in range(1, N):
        ok = True
        # The left part: for indices 0 .. k-1, boxes are taken from B[0..k-1]
        for i in range(0, k):
            if B[i] < A[i]:
                ok = False
                break
        if not ok:
            continue
        # The right part: for indices k+1 .. N-1 in the new box list D,
        # D[i] = B[i-1] for i in [k+1, N-1]
        for i in range(k+1, N):
            if B[i-1] < A[i]:
                ok = False
                break
        if not ok:
            continue
        # For the new inserted box at position k, we require:
        #   x >= A[k] and also, if k > 0, the preceding box B[k-1] <= x.
        # Thus x must be at least max(A[k], B[k-1]).
        candidate = max(A[k], B[k-1])
        ans = min(ans, candidate)
    
    # Case 3: Insertion at the end (position k = N, new box is the largest).
    # Then for i in 0..N-2, D[i] = B[i] and must satisfy B[i] >= A[i]
    ok = True
    for i in range(0, N-1):
        if B[i] < A[i]:
            ok = False
            break
    if ok:
        candidate = A[N-1]  # new box (largest) should be >= A[N-1]
        ans = min(ans, candidate)
    
    if ans == 10**18:
        print(-1)
    else:
        print(ans)
    
if __name__ == '__main__':
    main()