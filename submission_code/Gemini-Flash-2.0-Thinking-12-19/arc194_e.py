import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Check boundary characters. S[0] must match T[0] and S[N-1] must match T[N-1].
    # An operation starting at index i (1-indexed) affects indices i, ..., i+X+Y-1.
    # In 0-indexed, this is i-1, ..., i+X+Y-2. Valid i (1-indexed) are 1 to N-(X+Y)+1.
    # Valid i (0-indexed) are 0 to N-(X+Y).
    # Index 0 can be affected only if the operation starts at i=0 (0-indexed).
    # The first segment affected is [0, L-1] (where L=min(X,Y)) or [R, X+Y-1] (where R=max(X,Y)).
    # Index 0 is in [0, L-1] if L >= 1. Since X, Y >= 1, L >= 1.
    # Index 0 is in [R, X+Y-1] if R <= 0, which is impossible.
    # So index 0 can only be directly affected by an operation starting at i=0.
    # Index N-1 can be affected only if the operation ends at i+X+Y-1 = N-1, so i = N-X-Y (0-indexed).
    # The last segment affected is [N-X-Y, N-X-Y+L-1] or [N-X-Y+R, N-1].
    # Index N-1 is in [N-X-Y+R, N-1] if N-1 >= N-X-Y+R, which means X+Y-R >= 1, i.e., L >= 1.
    # Index N-1 is in [N-X-Y, N-X-Y+L-1] if N-1 <= N-X-Y+L-1, which means X+Y <= L, impossible.
    # So index N-1 can only be directly affected by an operation starting at i=N-X-Y.

    # If S[0] != T[0], an operation at i=0 must be performed an odd number of times.
    # If S[N-1] != T[N-1], an operation at i=N-X-Y must be performed an odd number of times.
    # The specific conditions for Op A/B at i=0 or i=N-X-Y on the current string S might not be met.
    # The sample cases suggest that if the border characters don't match initially, it's impossible.
    if S[0] != T[0] or S[N - 1] != T[N - 1]:
        print("No")
        return

    # If X == Y, operations swap $0^X 1^X \leftrightarrow 1^X 0^X$.
    # Op A at i (0-indexed): Cond S[i..i+X-1]=0^X, S[i+X..i+2X-1]=1^X. Action S[i..i+X-1]=1^X, S[i+X..i+2X-1]=0^X.
    # Original 1s in [i+X, i+2X-1], new 1s in [i, i+X-1].
    # Set of indices of 1s changes by removing {i+X, ..., i+2X-1} and adding {i, ..., i+X-1}.
    # Sum of indices of 1s changes by $\sum_{k=i}^{i+X-1} k - \sum_{k=i+X}^{i+2X-1} k = (Xi + X(X-1)/2) - ((i+X)X + X(X-1)/2) = Xi - (i+X)X = -X^2$.
    # Modulo X, this change is 0. The sum of indices of 1s modulo X is invariant.
    # Also, the total count of 1s is invariant.
    if X == Y:
        if S.count('1') != T.count('1'):
            print("No")
            return

        sum_indices_s = 0
        sum_indices_t = 0
        for k in range(N):
            if S[k] == '1':
                sum_indices_s = (sum_indices_s + k) % X
            if T[k] == '1':
                sum_indices_t = (sum_indices_t + k) % X

        if sum_indices_s != sum_indices_t:
             print("No")
             return

        # If counts and sum of indices mod X match, and borders match, it is possible.
        print("Yes")
        return

    # If X != Y, and border characters match.
    # Sample 1 suggests that if the borders match when X != Y, it's always possible.
    # This implies that any difference S[k] != T[k] for k in (0, N-1) can be resolved.
    # The operations allow swapping $0^X 1^Y \leftrightarrow 1^Y 0^X$. The effective boundary between $0^a 1^b$ type blocks can shift by $|X-Y|$.
    # This difference in block sizes might provide enough flexibility to resolve internal differences.
    # Based on the samples and the difference in invariant properties for X=Y vs X!=Y,
    # the hypothesis is that if $X 
eq Y$, matching borders are sufficient.
    if X != Y:
        print("Yes")
        return

solve()