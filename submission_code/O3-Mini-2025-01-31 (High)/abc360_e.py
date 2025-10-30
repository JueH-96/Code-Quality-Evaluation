def main():
    import sys
    mod = 998244353

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    
    # Special case: N = 1, the only ball is black so its position remains 1.
    if N == 1:
        sys.stdout.write("1")
        return

    # In the problem we have N balls (one black, N-1 white) arranged linearly.
    # At each operation, we pick two integers a and b uniformly from 1 to N.
    # If a != b, we swap the balls at positions a and b.
    #
    # Let the current black ball position be i.
    # In one operation:
    # - With probability N/N^2 (when a == b), nothing happens so i stays.
    # - With probability (N-1)*(N-2)/N^2 (when neither chosen equals i), nothing happens.
    # - But if one of the chosen positions is i, then the black ball gets swapped to the other position.
    #   For each j != i, there are exactly 2 outcomes (a = i, b = j and a = j, b = i) each with probability 1/N^2.
    #
    # Therefore the transition probabilities for the black ball from position i are:
    #   • P(remain at i) = (N (from a == b) + (N-1)(N-2)) / N^2 = (N^2 - 2N + 2) / N^2.
    #   • P(move to any specific j ≠ i) = 2/N^2.
    #
    # Thus, the expected new position given the black ball is at position i is:
    #   E[new position] = i * (N^2 - 2N + 2)/N^2 + (2/N^2) * (sum_{j=1..N} j - i)
    # Notice that sum_{j=1..N} j = N(N+1)/2.
    #
    # Simplifying:
    #   E[new] = i * (N^2 - 2N + 2)/N^2 + (2/N^2)*(N(N+1)/2 - i)
    #            = i * (N^2 - 2N + 2)/N^2 + (N(N+1))/N^2 - (2i)/N^2
    #            = i * (N^2 - 2N)/N^2 + (N+1)/N
    #            = i*(N-2)/N + (N+1)/N.
    #
    # Hence if we define:
    #   a = (N-2)/N   and   b = (N+1)/N,
    # then the recurrence for the expected position E_k after k operations is:
    #   E_{k+1} = a * E_k + b,  with initial E_0 = 1.
    #
    # The closed form solution of such a linear recurrence is:
    #   E_K = a^K * E_0 + b * (1 + a + a^2 + ... + a^(K-1))
    #       = a^K + b * ((1 - a^K)/(1 - a)) .
    #
    # We need to compute this expected value modulo mod.
    #
    # All divisions modulo mod are done using the modular inverse.
    
    invN = pow(N, mod - 2, mod)   # inverse of N modulo mod
    a = ((N - 2) % mod) * invN % mod   # a = (N-2)/N modulo mod
    b = ((N + 1) % mod) * invN % mod   # b = (N+1)/N modulo mod

    aK = pow(a, K, mod)  # compute a^K mod mod

    # Compute the series sum term: b * (1 - a^K) / (1 - a)
    # First, compute the inverse of (1 - a) modulo mod.
    inv1_minus_a = pow((1 - a) % mod, mod - 2, mod)
    series_term = (b * ((1 - aK) % mod)) % mod
    series_term = (series_term * inv1_minus_a) % mod

    result = (aK + series_term) % mod
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()