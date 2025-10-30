def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    N = int(data[0])
    s = data[1].strip()
    mod = 998244353

    # Explanation:
    # We are given a graph defined on vertices {0, 1, ..., N} with two types of edges:
    #    (1) A cycle among vertices 0 to N-1. For each i from 0 to N-1, there is an edge between
    #        i and (i+1) mod N.
    #    (2) For each i from 0 to N-1, if s[i]=='1', then there is an edge between i and N.
    #
    # Then for each undirected edge, we choose a direction.
    #
    # How do these choices affect the in-degrees?
    #   For the cycle edge between i and (i+1):
    #     - If we direct it as i -> (i+1), then vertex (i+1) gets +1.
    #     - If we direct it as (i+1) -> i, then vertex i gets +1.
    #
    #   For a star edge between i and N (only if s[i]=='1'):
    #     - We have a binary choice: either i -> N (then vertex N gets +1) or N -> i (then vertex i gets +1).
    #
    # Let us denote the choices on the cycle edges by a bit-array X of length N.
    # Define for i in 0..N-1:
    #    X[i] = 1 if the edge (i -> (i+1)) is chosen,
    #           0 if the edge ((i+1) -> i) is chosen.
    #
    # Then the contribution of the cycle edge to vertex degrees is:
    #   For vertex i (0 <= i <= N-1), it appears in exactly 2 cycle edges:
    #       - from the edge connecting (i-1) and i:
    #           contributes +1 if X[i-1] == 1. (Indices modulo N)
    #       - from the edge connecting i and (i+1):
    #           contributes +1 if X[i] == 0.
    #   So let A[i] = (1 if X[i-1]==1 else 0) + (1 if X[i]==0 else 0).
    #   For vertex N, there is no cycle edge.
    #
    # For a star edge at index i (only if s[i]=='1'), let Y[i] be a choice:
    #   - Y[i] = 1 means the star edge is directed as i -> N. 
    #         It contributes +1 to vertex N.
    #   - Y[i] = 0 means it is directed as N -> i.
    #         It contributes +1 to vertex i.
    #
    # Therefore, the final in-degrees become:
    #   For 0 <= i <= N-1:
    #       d[i] = A[i] + (if s[i]=='1' then (1 if Y[i]==0 else 0) else 0)
    #   And for vertex N:
    #       d[N] = sum_{i where s[i]=='1'} (1 if Y[i]==1 else 0)
    #
    # Notice that for each star edge (when s[i]=='1'), the total contribution to 
    # d[i] + d[N] is always 1 (either 0 and 1 or 1 and 0).
    # So the star choices simply allow us, for each i with s[i]=='1', to add an extra 0 or 1
    # to vertex i. This gives 2 independent choices per star edge.
    #
    # Now focus on the cycle part.
    # The cycle part is determined by X. Notice that for each edge (i, i+1), exactly one of
    # the two vertices gets +1. In particular, if X is the binary sequence for cycle choices,
    # then for each i (taking indices modulo N) we have:
    #      A[i] = 1 + (X[i-1] - X[i]),
    # which means A[i] is 2 if (X[i-1],X[i]) = (1,0), is 0 if (X[i-1],X[i]) = (0,1)
    # and is 1 if both bits are equal.
    #
    # A key observation is that while there are 2^N choices for X, two different X's might
    # yield the same A-vector. In fact, one may check (for example, when X is constant 0 or constant 1)
    # that the all-1 vector A arises from two X choices.
    #
    # In fact, one can prove that aside from the one collision when X is constant, the mapping X -> A
    # is one-to-one. Thus, the number of distinct A vectors is exactly 2^N - 1.
    # (For N=3, this gives 2^3 - 1 = 7 distinct cycle outcomes, matching the sample.)
    #
    # Finally, the overall d-sequence is completely determined by the cycle part (the A-vector, which is
    # a vector on vertices 0..N-1) and the star part (the independent binary choices for each i with s[i]=='1').
    #
    # Thus, the number of distinct in-degree sequences is:
    #     (# distinct cycle outcomes) * (2^(number of star edges)).
    #
    # Which is:
    #     (2^N - 1) * 2^(popcount(s))
    #
    # We compute that modulo 998244353.
    
    ones = s.count('1')
    cycle_count = (pow(2, N, mod) - 1) % mod
    star_count = pow(2, ones, mod)
    ans = (cycle_count * star_count) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()