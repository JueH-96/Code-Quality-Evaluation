import sys
import bisect

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    I_AB = [] # A_i >= 0, B_i >= 0
    I_A_ = [] # A_i >= 0, B_i == -1
    I__B = [] # A_i == -1, B_i >= 0
    I___ = [] # A_i == -1, B_i == -1

    for i in range(N):
        if A[i] >= 0 and B[i] >= 0:
            I_AB.append(i)
        elif A[i] >= 0 and B[i] == -1:
            I_A_.append(i)
        elif A[i] == -1 and B[i] >= 0:
            I__B.append(i)
        else:
            I___.append(i)

    k = len(I___)

    S = None
    if I_AB:
        S = A[I_AB[0]] + B[I_AB[0]]
        for i in I_AB:
            if A[i] + B[i] != S:
                print("No")
                return

    if S is None:
        # If I_AB is empty, S is not fixed by A_i, B_i >= 0 pairs.
        # S must be large enough for non-negativity: S >= A_i for i in I_A_, S >= B_i for i in I__B.
        # S must also be >= 0.
        min_S = 0
        for i in I_A_:
            min_S = max(min_S, A[i])
        for i in I__B:
            min_S = max(min_S, B[i])
        
        # If I_AB is empty, the multisets L and R constructed for any valid S (e.g., S=min_S)
        # are identical. This implies c_L(v) = c_R(v) for all v.
        # The conditions become |0| <= k (for v < S) and 0 <= 0 <= k (for v >= S),
        # which are true for any k >= 0. k is always >= 0.
        # We also need to ensure that elements in L (and R) are non-negative for S = min_S.
        # Elements in L are A_i for i in I_A_ (>=0 by definition) and min_S - B_i for i in I__B (>=0 by definition of min_S).
        # So, if I_AB is empty, a solution is always possible.
        print("Yes")
        return

    # S is fixed by I_AB

    # Check non-negativity constraints imposed by S
    # For A_i >= 0, B_i = -1: S must be >= A_i so B'_i = S - A_i >= 0
    for i in I_A_:
        if S < A[i]:
            print("No")
            return
    # For A_i = -1, B_i >= 0: S must be >= B_i so A'_i = S - B_i >= 0
    for i in I__B:
        if S < B[i]:
            print("No")
            return
    # For A_i >= 0, B_i >= 0 (in I_AB): S = A_i + B_i >= 0 is guaranteed

    # Construct multisets L and R
    # L: values required in modified A' from non-Type 4 indices
    # R: target values required in modified A' corresponding to modified B' from non-Type 4 indices
    L = []
    R = []

    for i in I_AB:
        L.append(A[i])
        R.append(S - B[i]) # which is A[i]

    for i in I_A_:
        L.append(A[i])
        R.append(S - (S - A[i])) # which is A[i]. B'_i for this index is S-A_i

    for i in I__B:
        L.append(S - B[i]) # A'_i for this index is S-B_i
        R.append(S - B[i]) # Target A' for this index is S-B'_i = S-B_i

    L.sort()
    R.sort()

    # Collect critical values for v where c_L(v) - c_R(v) can change
    # These are values in L, R, and S. Also check points just below these values.
    test_points = set()
    for val in L + R + [S]:
         test_points.add(val)
         test_points.add(val - 1)
    test_points.add(0) # Ensure 0 is checked, as non-negativity is key
    test_points = sorted(list(test_points))

    def count_le(sorted_list, val):
        return bisect.bisect_right(sorted_list, val)

    # Check conditions |c_R(v) - c_L(v)| <= k for v < S and 0 <= c_L(v) - c_R(v) <= k for v >= S
    for v in test_points:
        # Values in L and R are guaranteed to be non-negative based on the S checks above.
        # So c_L(v) = c_R(v) = 0 for v < 0.
        if v < 0:
             continue # c_L(v) = c_R(v) = 0, |0| <= k, 0 <= 0 <= k hold if k >= 0

        cLv = count_le(L, v)
        cRv = count_le(R, v)

        if v < S:
            # Condition: |c_R(v) - c_L(v)| <= k
            if abs(cRv - cLv) > k:
                print("No")
                return
        else: # v >= S
            # Condition: 0 <= c_L(v) - c_R(v) <= k
            if cLv < cRv or cLv - cRv > k:
                print("No")
                return

    print("Yes")

solve()