import sys

def solve():
    N = int(sys.stdin.readline())
    A = sys.stdin.readline().strip()
    B = sys.stdin.readline().strip()

    s = [i for i, c in enumerate(A) if c == '1']
    t = [i for i, c in enumerate(B) if c == '1']

    k = len(s)
    m = len(t)

    # If the number of initial pieces is less than the number of squares that must have pieces (m),
    # it is impossible to place k pieces such that each of the m required squares gets at least one piece.
    # The operation "moves all pieces" implies the total number of pieces is conserved.
    # The final configuration requires that squares with B_i=1 must have >= 1 piece, and squares with B_i=0 must have 0 pieces.
    # This means the set of occupied squares must be exactly the set of indices where B_i=1.
    # If k pieces end up only on squares in S_B (where B_i=1), and each square in S_B has >= 1 piece,
    # the minimum total number of pieces required is |S_B| = m.
    # Thus, if k < m, it's impossible.
    if m > k:
        print(-1)
        return

    # We need to transform the initial sorted positions s = [s_0, ..., s_{k-1}]
    # into a final sorted multiset p = [p_0, ..., p_{k-1}]
    # such that the set of unique values in p is exactly {t_0, ..., t_{m-1}}.
    # The minimum number of operations to transform sorted s to sorted p using this specific operation
    # is given by max(s_j - p_j) - min(s_j - p_j) over all j=0..k-1.
    # We need to find a valid target multiset p that minimizes this cost.

    # A sorted multiset p = [p_0, ..., p_{k-1}] has unique values {t_0, ..., t_{m-1}} if and only if:
    # 1. Every element p_j is in {t_0, ..., t_{m-1}}. This means t[0] <= p_j <= t[m-1] for all j.
    # 2. For every i in {0, ..., m-1}, t[i] is present in the multiset p.
    # These conditions translate to bounds on the sorted elements p_j:
    # p_j must be >= the (j - (k-m) + 1)-th smallest required position (t), if j-(k-m) is non-negative.
    # p_j must be <= the (j+1)-th smallest required position (t), if j < m.
    # Using 0-based indices for t and j:
    # p_j >= t[max(0, j - (k-m))] for j = 0..k-1  (Let this be L[j])
    # p_j <= t[min(m-1, j)] for j = 0..k-1      (Let this be U[j])

    # We need to find a sorted sequence p = [p_0, ..., p_{k-1}] such that L[j] <= p_j <= U[j] for all j,
    # and the set of unique values in p is {t_0, ..., t_{m-1}}, which minimizes max(s_j - p_j) - min(s_j - p_j).
    # The existence of a sorted sequence p satisfying L[j] <= p_j <= U[j] is equivalent to
    # max_{0<=i<=j} L[i] <= min_{j<=i<=k-1} U[i] for all j.
    # Let hat_L[j] = max_{0<=i<=j} L[i] and hat_U[j] = min_{j<=i<=k-1} U[i].
    # Any sorted sequence p satisfying L[j] <= p_j <= U[j] must also satisfy hat_L[j] <= p_j <= hat_U[j].
    # The sorted sequence p that satisfies hat_L[j] <= p_j <= hat_U[j] and minimizes max(s_j - p_j) - min(s_j - p_j)
    # is the one where p_j is chosen optimally within these bounds.
    # The minimum value of max(s_j - p_j) - min(s_j - p_j) over all sorted p satisfying hat_L[j] <= p_j <= hat_U[j]
    # is given by max_j(s_j - hat_U[j]) - min_j(s_j - hat_L[j]).
    # The paper confirms that the optimal p for the unique value property is also bounded by hat_L and hat_U,
    # making this formula applicable.

    L = [0] * k
    U = [0] * k
    for j in range(k):
        L[j] = t[max(0, j - (k - m))]
        U[j] = t[min(m - 1, j)]

    # Calculate prefix maximums of L
    hat_L = [0] * k
    hat_L[0] = L[0]
    for j in range(1, k):
        hat_L[j] = max(hat_L[j-1], L[j])

    # Calculate suffix minimums of U
    hat_U = [0] * k
    hat_U[k-1] = U[k-1]
    for j in range(k-2, -1, -1):
        hat_U[j] = min(hat_U[j+1], U[j])

    # Calculate max(s_j - hat_U_j) and min(s_j - hat_L_j)
    max_s_minus_hat_U = float('-inf')
    min_s_minus_hat_L = float('inf')

    for j in range(k):
        max_s_minus_hat_U = max(max_s_minus_hat_U, s[j] - hat_U[j])
        min_s_minus_hat_L = min(min_s_minus_hat_L, s[j] - hat_L[j])

    print(max_s_minus_hat_U - min_s_minus_hat_L)


T = int(sys.stdin.readline())
for _ in range(T):
    solve()