def main():
    import sys
    input = sys.stdin.readline

    # Read inputs
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))

    # If M == N, then all candidates are guaranteed to be elected
    # because "the number of candidates who have more votes" can never be N or more.
    # So everyone needs 0 additional votes.
    if M == N:
        print(" ".join(["0"] * N))
        return

    # Total votes used so far
    S = sum(A)
    # Remaining votes
    R = K - S

    # Pair up (votes, candidate_id) and sort descending by votes
    arr = [(A[i], i) for i in range(N)]
    arr.sort(key=lambda x: x[0], reverse=True)

    # C will be the sorted votes (descending), and idxOf[i] = position of candidate i in C
    C = [arr[i][0] for i in range(N)]
    idxOf = [0]*N
    for pos, (_, cand) in enumerate(arr):
        idxOf[cand] = pos

    # Build prefix sums of C for quick range-sum queries
    # prefixC[x] = sum of C[0] + C[1] + ... + C[x-1]
    prefixC = [0]*(N+1)
    for i in range(N):
        prefixC[i+1] = prefixC[i] + C[i]

    # A small helper: getPos(T) returns how many elements in C are strictly greater than T.
    # C is in descending order, so we do a binary search for the point
    # where elements become <= T.
    def getPos(T):
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi) // 2
            if C[mid] > T:
                lo = mid + 1
            else:
                hi = mid
        return lo  # number of elements strictly greater than T

    # Feasibility check:
    # Returns True if giving X extra votes to candidate i guarantees
    # that candidate i is in the top M (no matter how the other R - X votes are distributed).
    def feasible(i, X):
        T = A[i] + X  # final votes for candidate i

        # L0 = how many candidates already have more than T votes (strictly above i)
        p = getPos(T)
        L0 = p
        # If at least M candidates are already strictly above i, i cannot be top M
        if L0 >= M:
            return False

        # We need to see if "the others" can push (M - L0) more candidates above i.
        F = M - L0  # how many we must lift from <= T to surpass i
        # The sub of candidates who have votes <= T is everything from index p..N-1 in C
        # but we must exclude i itself if i appears in that range. That sub has size (N-p).
        # Excluding i gives subSize = (N - p) - 1
        subSize = (N - p) - 1
        # If they need F > subSize, they cannot push enough to get M surpassers => i is safe
        if F > subSize:
            return True

        # Otherwise, we try to pick the largest F from that sub to minimize the cost of pushing them above T.
        # The top F in that sub (descending) would be C[p], C[p+1], ..., C[p+F-1], but we must skip i's index if it lies there.
        end_ = p + F
        if end_ > N:  
            # not enough elements
            return True

        # Sum of the top F in that sub if we did not skip i:
        sumRange = prefixC[end_] - prefixC[p]

        posI = idxOf[i]
        # If candidate i's index is in the chunk [p, p+F-1], we skip it and replace it with the next best element C[p+F],
        # provided p+F < N. If p+F == N, then we can't pick F distinct others => i is safe => feasible = True.
        if p <= posI < end_:
            if end_ == N:
                # There's no "next candidate" beyond p+F-1
                return True
            # Remove i's votes from sumRange, add next largest from index end_
            sumRange = sumRange - C[posI] + C[end_]

        # cost = total votes "others" need to push those F above T
        cost = F * (T + 1) - sumRange
        # If cost <= R - X, then it's possible to push those F above i => i not guaranteed
        # We want cost > R - X to be guaranteed.
        return cost > (R - X)

    # We'll answer for each candidate i
    ans = [0]*N

    for i in range(N):
        # Quick check if i is already guaranteed with X=0
        if feasible(i, 0):
            ans[i] = 0
            continue

        # If even giving all R votes to i is not enough, answer is -1
        if not feasible(i, R):
            ans[i] = -1
            continue

        # Otherwise, binary search in [1..R] to find the minimal X
        lo, hi = 1, R
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(i, mid):
                hi = mid
            else:
                lo = mid + 1
        ans[i] = lo

    print(" ".join(map(str, ans)))