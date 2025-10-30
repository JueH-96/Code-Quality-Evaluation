def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Parse inputs
    N = int(data[0])
    s = data[1].strip()
    mod = 998244353

    # Explanation:
    # We are given a graph with N cycle edges and additional edges from vertices i with s[i]=='1'
    # to an extra vertex (vertex number N). When we assign directions,
    # the in‐degree from the cycle for vertex i (with 0 ≤ i < N) is determined as follows.
    #
    # Label the N cycle edges by a binary sequence a = (a0,...,a_{N-1}) ∈ {0,1}^N.
    # For each i, interpret a_i as:
    #   • if a_i == 0, direct the edge from vertex i to vertex (i+1) mod N.
    #   • if a_i == 1, direct the edge from vertex (i+1) mod N to vertex i.
    #
    # Then the in‐degree contribution from the cycle to vertex i is given by:
    #    X_i = (1 if the edge from vertex (i-1) mod N is directed from (i-1) to i else 0)
    #          + (1 if the edge from vertex i to vertex (i+1) is directed from (i+1) to i else 0).
    #
    # Notice that if we write this using the bits of a, we get:
    #    X_i = (1 if a_{i-1} == 0 else 0) + (1 if a_i == 1 else 0)
    # which can be rewritten as X_i = 1 + a_i - a_{i-1} (with indices mod N).
    #
    # A key observation is that the map f from a to the cycle‐degree sequence X = (X_0,...,X_{N-1})
    # is one‐to‐one for every nonconstant a. However, the two constant sequences a = 000…0 and
    # a = 111…1 both yield X = (1, 1, …, 1). Thus, although there are 2^N total choices for a,
    # the number of distinct cycle in‐degree sequences is (2^N - 1).
    #
    # Now, for each extra edge (i.e. each i with s[i]=='1'):
    #  • We decide the orientation. If the edge is oriented from i to N, vertex N gets +1.
    #  • Otherwise (if oriented from N to i) vertex i gets a bonus of +1.
    # That is, for each such i we have a binary choice that modifies the in‐degree at i and contributes
    # to the in‐degree of vertex N. There are exactly 2^(popcount(s)) possibilities.
    #
    # Finally, these two parts (the cycle and the extra edges) are independent.
    # Hence the total number of distinct in‐degree sequences is:
    #
    #      (2^N - 1) * 2^(number of ones in s)
    #
    # We compute the answer modulo 998244353.
    
    count1 = s.count('1')
    cycle_count = (pow(2, N, mod) - 1) % mod  # distinct outcomes from cycle orientations
    extra_count = pow(2, count1, mod)          # outcomes from extra edge orientations
    ans = (cycle_count * extra_count) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()