import sys

def solve():
    N = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    
    C = s.count('1')
    
    MOD = 998244353
    
    # The problem asks for the number of distinct in-degree sequences (d_0, d_1, ..., d_N).
    # The graph G has vertices 0, ..., N.
    # Edges are: {i, (i+1)%N} for i=0,...,N-1 (cycle) and {i, N} if s_i=1 (star).
    # Let c_i in {0, 1} be the orientation of edge {i, (i+1)%N}.
    # Let e_i in {0, 1} be the orientation of edge {i, N} for i where s_i=1.
    # d_i for i in {0, ..., N-1} depends on cycle edges involving i and star edge {i, N} if s_i=1.
    # d_i = (1-c_{i-1}) + c_i + (1-e_i)s_i (indices mod N)
    # d_N depends only on star edges: d_N = sum_{i=0}^{N-1} e_i * s_i.

    # Let d'_i = (1-c_{i-1}) + c_i for i in {0, ..., N-1}. This is the in-degree from cycle edges.
    # The set of achievable sequences (d'_0, ..., d'_{N-1}) by choosing c in {0,1}^N is denoted D'.
    # The size of D' is 2^N - 1 for N >= 2.
    # Let a_i = (1-e_i)s_i for i in {0, ..., N-1}. This is the in-degree contribution from star edges to vertices 0..N-1.
    # If s_i=0, a_i = 0. If s_i=1, a_i = 1-e_i which is in {0, 1}.
    # The set of achievable sequences (a_0, ..., a_{N-1}) by choosing e_i for i with s_i=1 is denoted A.
    # The size of A is 2^C, where C is the number of 1s in s.
    
    # We have (d_0, ..., d_{N-1}) = d' + a, where d' in D' and a in A.
    # d_N = sum_{i:s_i=1} e_i = sum_{i:s_i=1} (1 - (1-e_i)) = C - sum_{i:s_i=1} (1-e_i) = C - sum_{i=0}^{N-1} a_i.
    # So, the sequence a uniquely determines d_N.
    # The set of distinct (d_0, ..., d_N) sequences is the set of (d'+a, C-sum(a_i)) for d' in D', a in A.
    # This set is in one-to-one correspondence with the set of distinct (d'+a) sequences,
    # because d'+a uniquely determines sum(a_i) = sum(d'+a) - sum(d') = sum(d'+a) - N, which determines d_N.
    # The number of distinct sequences is the size of the sumset D' + A = {d'+a | d' in D', a in A}.

    # The size of the sumset |X+Y| = sum_{y in Y} |(X+y) \ (cup_{y'<y} (X+y'))|.
    # A simpler formula is |X+Y| = |X| + |Y| - |X \cap Y| if X, Y are vector spaces (or subgroups) and the sum is direct.
    # Here D' and A are just subsets of Z^N.
    # The size of the sumset |D' + A| = |D'| * |A| if and only if (D'-D') intersection (A-A) = {0}.

    # D'-D' = {d1'-d2' | d1', d2' in D'}. These differences have sum 0.
    # A-A = {a1-a2 | a1, a2 in A}. alpha = a1-a2 has alpha_i = 0 if s_i=0, alpha_i in {-1,0,1} if s_i=1.
    # If s has a cyclic substring "00" or "11", then (D'-D') intersection (A-A) is {0}.
    # If s is all '0's, C=0, A={0}, A-A={0}. Intersection is {0}.
    # If s is not all '0's, check for "00" or "11" cyclically.
    
    has_00_or_11_cyclic = False
    if N > 1:
        for i in range(N):
            if s[i] == s[(i+1)%N]:
                has_00_or_11_cyclic = True
                break

    if has_00_or_11_cyclic or C == 0: # C=0 means s is all 0s, which has cyclic "00" unless N=1. N>=3.
        # In this case, (D'-D') intersection (A-A) is {0}.
        # The size of the sumset |D' + A| = |D'| * |A| = (2^N - 1) * 2^C.
        # For N=1, cycle graph is {0,0}. Problem constraint N>=3.
        ans = ((pow(2, N, MOD) - 1 + MOD) * pow(2, C, MOD)) % MOD
    else:
        # If s is not all '0's and has no cyclic "00" or "11", s must be alternating "0101..." or "1010...".
        # This is only possible if N is even and C = N/2.
        # In this specific case, (D'-D') intersection (A-A) is non-trivial.
        # The size of D'+A is known to be 2^N + 2^C - 2.
        ans = (pow(2, N, MOD) + pow(2, C, MOD) - 2 + MOD) % MOD

    print(ans)

solve()