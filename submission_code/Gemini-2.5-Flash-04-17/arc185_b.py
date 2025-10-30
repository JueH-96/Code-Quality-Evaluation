import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Compute prefix sums S_k = sum(A_1 to A_k)
    S = [0] * (N + 1)
    for i in range(N):
        S[i+1] = S[i] + A[i]

    # We need to find if there exists a sequence of prefix sums T_k for a non-decreasing sequence B
    # such that T_0 = 0, T_N = S_N, T_k >= S_k for 1 <= k < N (reachability),
    # and T_k - T_{k-1} is non-decreasing (B_k >= B_{k-1}), which means T_k is convex.
    # T_k >= T_{k-1} and T_k - T_{k-1} >= T_{k-1} - T_{k-2} for k >= 2
    # T_k >= 2 * T_{k-1} - T_{k-2} for k >= 2
    # Also B_1 >= 0 => T_1 >= T_0 = 0. Since A_1 >= 0, S_1 >= 0. Reachability T_1 >= S_1 covers T_1 >= 0.

    # Conditions on T_k:
    # 1. T_0 = 0
    # 2. T_N = S_N
    # 3. T_k >= S_k for 1 <= k <= N-1
    # 4. T_k >= 2 * T_{k-1} - T_{k-2} for 2 <= k <= N

    # Let M_k be the minimum possible value for T_k satisfying T_0=0, T_i >= S_i (i < k),
    # and T_i >= 2*T_{i-1} - T_{i-2} (2 <= i <= k).
    # M_0 = 0
    # M_1 >= S_1. Also M_1 >= 2 * M_0 - M_{-1} doesn't apply. Minimum M_1 should be S_1.
    # M_k >= S_k for k in [1, N-1]
    # M_k >= 2 * M_{k-1} - M_{k-2} for k in [2, N]

    # Let's try computing M_k recursively based on previous M values and S_k.
    # This construction gives the minimum sequence of prefix sums satisfying T_0=0 and convexity.
    # Let's call this sequence M_k.
    # M_0 = 0
    # M_1 = S_1 # T_1 must be >= S_1. Also T_1 >= 0. S_1 >= 0 covers this.
    # For k >= 2: M_k must be >= S_k (if k < N) and >= 2 * M_{k-1} - M_{k-2}.
    # Let's compute M_k for k = 0 to N.
    # M_0 = 0
    # M_1 = S[1] # Min T_1 must be >= S_1. T_1 >= 0 also required, but S_1 >= 0.
    # M_k = max(S[k], 2 * M[k-1] - M[k-2]) for k >= 2?

    # Let's define M_k as the minimum T_k satisfying T_0=0, T_i >= S_i for 1<=i<k, and T_i >= 2*T_{i-1}-T_{i-2} for 2<=i<=k.
    # M_0 = 0
    # M_1 = S[1] # T_1 >= S_1 is required.
    # For k = 2 .. N-1: M_k must be >= S_k and >= 2*M_{k-1} - M_{k-2}. Minimum M_k is max of these.
    # M_k = max(S[k], 2 * M_{k-1} - M_{k-2}) for k=2..N-1.

    # At k=N: T_N must be S_N and T_N >= 2*T_{N-1} - T_{N-2}.
    # So S_N >= 2*T_{N-1} - T_{N-2}.
    # For a valid sequence T_k to exist, we must be able to choose T_{N-1}, T_{N-2} ... T_0
    # satisfying the constraints up to N-1, such that S_N >= 2*T_{N-1} - T_{N-2}.
    # The minimum value of 2*T_{N-1} - T_{N-2} subject to constraints up to N-1 is 2*M_{N-1} - M_{N-2},
    # where M_0..M_{N-1} is the minimum sequence satisfying constraints up to N-1.

    M = [0] * (N + 1)
    M[0] = 0
    # M[1] must be >= S[1]. T_1 >= 0 is also required for B_1 >= 0. S[1] >= 0.
    # So M[1] must be >= S[1].
    # M[1] must also allow subsequent M[k] to satisfy constraints.
    # The minimum possible value for T_1 that allows a valid sequence?
    # This is where the Sample 2 (9,0) fails with M[1]=S[1]=9.
    # If T_1 can be smaller, e.g., 0, then T=(0,0,9) for (9,0) sum=9, non-decreasing.
    # T_0=0, T_1=0, T_2=9. T_2=S_2. T_1=0 < S_1=9. This T sequence is NOT reachable.

    # The reachability constraint is T_k >= S_k for k < N, T_N = S_N.
    # The non-decreasing B_k constraint is T_0=0, T_k is convex.
    # T_k >= 2*T_{k-1} - T_{k-2} for k >= 2, and T_1 >= 0.

    # Combining all conditions:
    # T_0 = 0
    # T_N = S_N
    # T_k >= S_k for 1 <= k <= N-1
    # T_k >= 2 * T_{k-1} - T_{k-2} for 2 <= k <= N
    # T_1 >= 0

    # Let M_k be the minimum possible value for T_k satisfying T_0=0, T_i >= S_i for 1<=i<k, T_i >= 2*T_{i-1}-T_{i-2} for 2<=i<=k, and T_1 >= 0.
    # M_0 = 0
    # M_1 >= 0. Also M_1 must allow later T_k >= S_k.
    # M_k >= S_k for k in [1, N-1]
    # M_k >= 2 * M_{k-1} - M_{k-2} for k in [2, N]
    # M_1 >= 0.

    # Let's try the Z_k logic again, which sets T_1 = 0 initially.
    # Z_0 = 0
    # Z_1 = 0 # Minimum T_1 >= 0
    # Z_k = max(S[k], 2 * Z[k-1] - Z[k-2]) for k=2..N-1 (satisfying T_k >= S_k (k<N) and convexity up to k)

    # Need to find if there exists T_0..T_N satisfying the conditions.
    # T_N = S_N.
    # T_0=0, T_k >= S_k (k<N), T_k convex (k>=2), T_1 >= 0.

    # Let's define M_k as the minimum prefix sum value at step k that satisfies
    # T_0=0, T_1 >= 0, T_i >= 2*T_{i-1} - T_{i-2} (2<=i<=k).
    # M_0 = 0
    # M_1 = 0 # Minimum value for T_1 >= 0
    # M_k = max(0, 2 * M_{k-1} - M_{k-2}) # Minimum value for T_k >= 2*T_{k-1}-T_{k-2} and T_k >= 0
    # Wait, T_k >= 2*T_{k-1}-T_{k-2} already implies T_k >= T_{k-1} if T_{k-1} >= T_{k-2}.
    # Base case M_0=0, M_1=0. M_2=0, M_3=0 ... M_k=0. This is T=(0,0,..,0), B=(0,0,..,0). Sum=0. Only possible if S_N=0.

    # The conditions are:
    # T_0 = 0
    # T_N = S_N
    # T_k >= S_k for 1 <= k <= N-1
    # T_k >= 2 * T_{k-1} - T_{k-2} for 2 <= k <= N

    # Let's try to construct the minimum T_k sequence that satisfies T_0=0, T_k >= S_k for k < N, and T_k is convex.
    # Let this sequence be M_k.
    # M_0 = 0
    # M_1 must be >= S[1]. Also must allow subsequent points to be >= S_k and convex.
    # M_k = max(S[k], 2 * M[k-1] - M[k-2]) for k = 1 .. N (S[N] is used here) seems promising but failed samples.
    # Let's redefine M_k:
    # M_0 = 0
    # M_1 = S[1] # T_1 must be >= S_1. This is the tightest lower bound from S.
    # For k = 2..N-1: T_k >= S_k and T_k >= 2*T_{k-1} - T_{k-2}. Minimum T_k is max of these.
    # M_k = max(S[k], 2 * M[k-1] - M[k-2]) for k=2..N-1.
    # At k=N: T_N must be S_N and T_N >= 2*T_{N-1} - T_{N-2}.
    # So we need S_N >= 2*T_{N-1} - T_{N-2}.
    # To minimize RHS, we must pick T_{N-1}, T_{N-2} ... T_0 to be minimal while satisfying constraints up to N-1.
    # The minimum sequence up to N-1 is M_0..M_{N-1}.
    # So we need to check if S_N >= 2*M_{N-1} - M_{N-2}.

    # M is computed using M[0]=0, M[1]=S[1], M[k]=max(S[k], 2*M[k-1]-M[k-2]) for k=2..N-1.
    # Need to use long long for M values.
    M = [0] * (N + 1)
    M[0] = 0
    if N >= 1:
        M[1] = S[1]
    for k in range(2, N):
        M[k] = max(S[k], 2 * M[k-1] - M[k-2])

    # Now check the condition at N.
    # We need S_N >= 2 * T_{N-1} - T_{N-2}.
    # The minimum value of the RHS is 2 * M[N-1] - M[N-2].
    # This is possible if S[N] >= 2 * M[N-1] - M[N-2].
    # Note: Need M[N-2] for N=2 case. M[N-2] is M[0].
    # For N=2: check S[2] >= 2 * M[1] - M[0].
    # For N=3: check S[3] >= 2 * M[2] - M[1].
    # For N=4: check S[4] >= 2 * M[3] - M[2].

    if N == 2:
        required_SN = 2 * M[1] - M[0]
    else: # N >= 3
        required_SN = 2 * M[N-1] - M[N-2]

    if S[N] >= required_SN:
        print("Yes")
    else:
        print("No")

T = int(sys.stdin.readline())
for _ in range(T):
    solve()