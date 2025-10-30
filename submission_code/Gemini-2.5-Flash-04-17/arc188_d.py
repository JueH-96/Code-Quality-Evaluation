import sys

def solve():
    MOD = 998244353

    def inv(x):
        return pow(x, MOD - 2, MOD)

    def factorial(n):
        res = 1
        for i in range(2, n + 1):
            res = (res * i) % MOD
        return res

    def combinations(n, k, fact, invFact):
        if k < 0 or k > n:
            return 0
        return (((fact[n] * invFact[k]) % MOD) * invFact[n - k]) % MOD

    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Precompute factorials and inverse factorials
    fact = [1] * (N + 1)
    invFact = [1] * (N + 1)
    for i in range(2, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    invFact[N] = inv(fact[N])
    for i in range(N - 1, 1, -1):
        invFact[i] = (invFact[i + 1] * (i + 1)) % MOD

    # Precompute D(n, k) = number of derangements of {1, ..., n}
    # with exactly k values i such that pi(i) > i.
    # This is equal to the number of permutations of {1, ..., n} with k ascents (i.e., pi(i) > i)
    # restricted to derangements.
    # The total number of permutations of {1, ..., n} with k ascents (i.e., pi(i) > i) is T(n, k).
    # T(n, k) is the coefficient of x^k in \sum_{j=0}^n \binom{n}{j} D_{n-j} x^j.
    # T(n, k) = A(n, n-k), where A(n, k) is the k-th Eulerian number related to descents i > pi(i+1).
    # A recurrence for T(n, k) (number of permutations of n elements with k elements i s.t. pi(i)>i)
    # is T(n, k) = k T(n-1, k) + (n-k+1) T(n-1, k-1) for n>=1, k>=1. T(n, 0) = 1 for n>=0. T(0,0)=1.

    # Recurrence for D(n, k): Number of derangements of n with k ascents (i < pi(i)).
    # A permutation of {1..n} with k ascents is either a derangement or has fixed points.
    # D(n, k) = T(n, k) - (number of permutations with >=1 fixed point and k ascents)
    # Number of permutations of n with j fixed points and k ascents is \binom{n}{j} D(n-j, k-j).
    # This seems complicated. A direct recurrence for D(n, k):
    # Consider element n.
    # Case 1: pi(n) = i, pi(i) = n for i != n. (n is in a 2-cycle).
    # There are n-1 choices for i.
    # If i < n, then n > pi(n)=i (descent), i < pi(i)=n (ascent). Contribution to ascents: 1.
    # The other n-2 elements form a derangement on {1..n-2} with k-1 ascents. There are D(n-2, k-1) such derangements. (n-1-k) choices for i < n. k-1 ascents needed.
    # If i > n, not possible as values are 1..n.
    # The number of elements i < n with pi(i) > i is k-1.
    # Recurrence: D(n, k) = k D(n-1, k) + (n-k+1) D(n-1, k-1) ...

    # Number of permutations of {1, ..., n} with exactly k values i s.t. pi(i) > i
    # Let this be T(n, k). T(n, k) is coefficient of x^k in \sum_{j=0}^n \binom{n}{j} D_{n-j} x^j.
    # D_n = n! sum_{j=0}^n (-1)^j / j!
    # T(n, k) = \sum_{j=0}^k \binom{n}{j} D_{n-j} \binom{n-j}{k-j}. No.

    # Let's use the provided recurrence for D(n, k) directly from OEIS A000528,
    # adjusted for derangements. Number of derangements of n elements with k ascents (i < pi(i))
    # T(n, k) = k T(n-1, k) + (n-k+1) T(n-1, k-1) with T(n, 0)=1, T(n, n)=0 for n>0.
    # T(n, k) is coefficient of x^k in \sum_{j=0}^n \binom{n}{j} D_{n-j} x^j.
    # D(n, k) = T(n, k) - \sum_{j=1}^n \binom{n}{j} D(n-j, k-j).
    # D(n, k) = T(n, k) - \binom{n}{1} D(n-1, k-1) - \binom{n}{2} D(n-2, k-2) - ...
    # D(n, k) = T(n, k) - n D(n-1, k-1) - \binom{n}{2} D(n-2, k-2) ...

    # Let's compute T(n, k) first.
    T = [[0 for _ in range(N + 1)] for _ in range(N + 1)]
    T[0][0] = 1
    for n in range(1, N + 1):
        T[n][0] = 1
        for k in range(1, n + 1):
            T[n][k] = (k * T[n - 1][k] + (n - k + 1) * T[n - 1][k - 1]) % MOD

    # Number of derangements D_n
    D_n = [0] * (N + 1)
    D_n[0] = 1
    if N >= 1: D_n[1] = 0
    for n in range(2, N + 1):
        D_n[n] = ((n - 1) * (D_n[n - 1] + D_n[n - 2])) % MOD

    # D(n, k) = number of derangements of {1..n} with k ascents (i < pi(i))
    # D(n, k) = \sum_{j=0}^n (-1)^j \binom{n}{j} T(n-j, k)
    # T(m, k) defined here is #permutations of m with k ascents (pi(i)>i)
    # The sum should be over number of fixed points.
    # Number of permutations of n with j fixed points and k ascents is \binom{n}{j} D(n-j, k - count of ascents among fixed points).
    # fixed points i: pi(i)=i. i < pi(i) is false. So fixed points don't contribute to k ascents.
    # D(n, k) = \sum_{j=0}^n (-1)^j \binom{n}{j} T(n-j, k).

    Dnk = [[0 for _ in range(N + 1)] for _ in range(N + 1)]
    for n in range(N + 1):
        for k in range(n + 1):
            # Calculate D(n, k) using inclusion-exclusion based on fixed points
            # D(n, k) = Sum over j fixed points * (-1)^j / j! * (perms of n-j with k ascents on n-j elements)
            # Number of permutations of n with j fixed points is $\binom{n}{j} D_{n-j}$
            # The number of permutations of n with j fixed points is $\binom{n}{j} \times (\text{number of permutations of } n-j)$.
            # Let $S_{n,k}$ be number of permutations of $n$ with $k$ ascents ($i < \pi(i)$). $S_{n, k} = T(n, k)$ as computed above.
            # A permutation of $n$ elements with $k$ ascents is formed by choosing $j$ fixed points
            # and a derangement of $n-j$ elements with $k$ ascents.
            # Sum_{j=0}^n \binom{n}{j} D(n-j, k) = T(n, k).
            # We can invert this using inclusion-exclusion.
            # D(n, k) = \sum_{j=0}^n (-1)^j \binom{n}{j} T(n-j, k)
            for j in range(n + 1):
                if k <= n - j:
                     term = (combinations(n, j, fact, invFact) * T[n - j][k]) % MOD
                     if j % 2 == 1:
                         Dnk[n][k] = (Dnk[n][k] - term + MOD) % MOD
                     else:
                         Dnk[n][k] = (Dnk[n][k] + term) % MOD

    # Identify sets and counts
    I_fixed = [i for i in range(N) if B[i] != -1]
    I_free = [i for i in range(N) if B[i] == -1]
    k_fixed = len(I_fixed)
    m_free = len(I_free)

    S_A = set(A)
    S_B = set(B[i] for i in I_fixed)
    U = set(range(1, 2 * N + 1)) - S_A - S_B

    S_A_free = sorted([A[i] for i in I_free])
    U_sorted = sorted(list(U))

    k_less_fixed = sum(1 for i in I_fixed if A[i] < B[i])
    k_greater_fixed = sum(1 for i in I_fixed if A[i] > B[i])

    # Compute G(m, k): # bijections from S_A_free to U_sorted with k ascents ($a'_i < u'_{\sigma(i)}$)
    # dp[i][j][k]: # bijections from {a'_1, ..., a'_i} to a size-i subset of {u'_1, ..., u'_j} with k ascents
    # a'_1 < ... < a'_m, u'_1 < ... < u'_m
    # dp[i][j][k] = ways to map first i elements of S_A_free to i elements from first j elements of U
    # Consider u'_j.
    # 1. u'_j is not in the image of {a'_1, ..., a'_i}. Map {a'_1, ..., a'_i} to size i subset of {u'_1, ..., u'_{j-1}}.
    #    Number of ways: dp[i][j-1][k].
    # 2. u'_j is in the image. Some a'_p maps to u'_j. Since a' are sorted, if a'_i maps to u'_j, it contributes k - (a'_i < u'_j) from first i-1 a' values mapped to first j-1 u' values.
    #    If a'_i maps to u'_j: map {a'_1, ..., a'_{i-1}} to size i-1 subset of {u'_1, ..., u'_{j-1}}. dp[i-1][j-1][k - (a'_i < u'_j)]
    #    This doesn't capture all possibilities. Any of a'_1 to a'_i could map to u'_j.
    # Correct DP for G(m,k): dp[i][j]: ways to map first i elements of S_A_free to *some* subset of U of size i, using *only* first j elements of U as potential images, having k ascents.
    # dp[i][j][k] = dp[i][j-1][k] (a_1...a_i map to subset of u_1...u_{j-1}) +
    #             dp[i-1][j-1][k - (a_i < u_j)] (a_i maps to u_j, rest map to subset of u_1...u_{j-1})
    # This counts number of ways to select i elements from u_1...u_j and form bijection with a_1...a_i.

    # Correct DP state for G(m, k): dp[i][j] = number of partial bijections from {a'_1, ..., a'_i}
    # to a subset of size $i$ from $\{u'_1, \dots, u'_j\}$ with $k$ ascents.
    # $dp[i][j][k]$ = $dp[i][j-1][k]$ ( $u'_j$ is not used as image) +
    #                $dp[i-1][j-1][k - (a'_i < u'_j)] \times i$ (some $a'_p$ maps to $u'_j$).
    # It should be $dp[i-1][j-1][k - (a'_i < u'_j)]$ if $a'_i$ maps to $u'_j$ (as $a'_1 \ldots a'_{i-1}$ maps to subset of $u'_1 \ldots u'_{j-1}$).

    G_mk = [[0 for _ in range(m_free + 1)] for _ in range(m_free + 1)]
    # G_mk[i][k]: number of bijections from {a'_1, ..., a'_i} to {u'_1, ..., u'_i} with k ascents.
    # This can be computed using inclusion-exclusion on pairs (a'_p, u'_q) where a'_p < u'_q.
    # Or using DP on ranks. $dp[i][j]$: number of bijections from $a'_1..a'_i$ to a subset of $u'_1..u'_j$ of size $i$.

    # Let $dp[i][j][k]$ be the number of ways to match elements $\{a'_1, \dots, a'_i\}$ to a subset of $\{u'_1, \dots, u'_j\}$ of size $i$ such that there are $k$ ascending pairs $(a'_p, u'_{q})$ where $a'_p < u'_q$.
    # $dp[i][j][k] = dp[i][j-1][k]$ ($u'_j$ is not used) + $dp[i-1][j-1][k - (a'_i < u'_j)]$ ($a'_i$ is matched with $u'_j$) + $dp[i][j-1][k - (a'_p < u'_j)]$ (some $a'_p, p<i$ is matched with $u'_j$)
    # This is not for bijections.

    # $G(m, k)$ can be computed with DP: $dp[i][j]$: number of permutations of first $i$ elements of $S_{A,free}$ mapped to first $j$ elements of $U$.
    # Correct DP for $G(m,k)$: $dp[i][j]$: # of ways to match $\{a'_1, \dots, a'_i\}$ to a size $i$ subset of $\{u'_1, \dots, u'_j\}$ with $k$ ascents.
    # $dp[i][j][k] = dp[i][j-1][k] + dp[i-1][j-1][k-(a'_i < u'_j)]$. Base cases $dp[0][j][0] = 1$.
    dp_G = [[[0 for _ in range(m_free + 1)] for _ in range(m_free + 1)] for _ in range(m_free + 1)]
    dp_G[0][0][0] = 1
    for i in range(m_free + 1):
        for j in range(m_free + 1):
            if i > j: continue
            if i == 0 and j == 0: continue
            if j > 0:
                for k in range(i + 1):
                    dp_G[i][j][k] = dp_G[i][j-1][k] # u'_j is not used
            if i > 0 and j > 0:
                 for k in range(i):
                    is_ascent = (S_A_free[i-1] < U_sorted[j-1])
                    if k - is_ascent >= 0:
                        dp_G[i][j][k] = (dp_G[i][j][k] + dp_G[i-1][j-1][k - is_ascent]) % MOD

    # G(m, k) is dp_G[m][m][k]

    total_sum_D = 0
    for k_free_ascents in range(m_free + 1):
        num_bijections_G = dp_G[m_free][m_free][k_free_ascents]
        if num_bijections_G == 0:
            continue
        # Number of free descents is m_free - k_free_ascents
        k_free_descents = m_free - k_free_ascents

        # Total descents = fixed descents + free descents
        total_descents = k_greater_fixed + k_free_descents

        # Total ascents = N - total_descents
        # total_ascents = k_less_fixed + k_free_ascents

        # Number of pairs of permutations (p1, p3) with total_descents descents (pi(v) > v)
        # and total_ascents ascents (v < pi(v)), no fixed points (pi(v)=v).
        # This is D(N, total_ascents) = D(N, total_descents).
        # D(N, k) = D(N, N-k).
        # N - total_descents = N - (k_greater_fixed + m_free - k_free_ascents)
        # = (k_less_fixed + k_greater_fixed + m_free) - k_greater_fixed - m_free + k_free_ascents
        # = k_less_fixed + k_free_ascents.
        # This is the number of total ascents.
        num_pairs_p1p3_D = Dnk[N][total_descents]

        term = (num_bijections_G * num_pairs_p1p3_D) % MOD
        total_sum_D = (total_sum_D + term) % MOD

    final_count = (fact[N] * total_sum_D) % MOD

    print(final_count)

solve()