def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    mod = 998244353

    # Special case: when there is only one ball the black ball must always be at position 1.
    if N == 1:
        sys.stdout.write("1")
        return

    # The process: There are N balls and a single black ball initially at position 1.
    # In one operation two indices a and b (from 1 to N) are chosen uniformly at random.
    # If a != b, the balls at these positions are swapped.
    #
    # Let the current position of the black ball be i.
    # It remains at i if neither chosen index is i, or if both indices are i.
    # Otherwise, if exactly one of the chosen indices is i then the black ball moves to the other index.
    #
    # Unconditionally, the chance that the black ball moves from i to any j ≠ i is:
    #   P(i -> j) = (for a = i, b = j) + (for b = i, a = j) = 1/N * 1/N + 1/N * 1/N = 2/N^2.
    # And the chance of staying in place i is:
    #   P(i -> i) = ((N-1)/N)^2 + 1/N^2 = (N^2 - 2N + 1 + 1)/N^2 = (N^2 - 2N + 2)/N^2.
    #
    # Let E_k be the expected position of the black ball after k operations.
    # Then, conditioned on the current position i,
    #    E(new position | i) = i * (N^2-2N+2)/N^2 + (2/N^2)*( (sum_{j=1}^N j) - i ).
    #
    # Since sum_{j=1}^N j = N(N+1)/2,
    # we can derive:
    #    E(new position | i) = [i*(N^2 - 2N) + N(N+1)] / N^2.
    #
    # Taking expectation over i leads us to the recurrence:
    #    E_{k+1} = A * E_k + B,
    # where A = (N^2 - 2N) / N^2 = 1 - 2/N, and B = N(N+1)/N^2 = (N+1)/N.
    #
    # For a recurrence of the form:
    #    E_{k+1} = A * E_k + B,  with E0 = 1,
    # the closed-form solution is:
    #    E_k = A^k + (B/(1 - A))*(1 - A^k).
    # Here (1 - A) = 2/N, so:
    #    E_k = A^k + ((N+1)/N)/(2/N) * (1 - A^k)
    #         = A^k + ((N+1)/2)*(1 - A^k)
    #         = (N+1)/2 + (1 - (N+1)/2)* A^k.
    #
    # Notice that 1 - (N+1)/2 = (1-N)/2.
    # Thus, the final expected position is:
    #    E_K = ((N+1) + (1-N)*A^K) / 2,   with A = 1 - 2/N.
    #
    # We now need to compute the answer modulo mod.
    # Since N may be large and mod is prime, we express division modulo mod
    # by using modular inverses.

    # Compute inverse of N mod mod.
    invN = pow(N, mod - 2, mod)
    # Compute A in modular arithmetic: A = 1 - 2/N.
    A = (1 - 2 * invN) % mod
    # Compute A^K mod mod.
    AK = pow(A, K, mod)
    # Expected position in rational form: ((N+1) + (1-N)*AK) / 2.
    # Compute numerator modulo mod.
    num = ((N + 1) + (1 - N) * AK) % mod
    # Compute inverse of 2 mod mod. Since mod is prime, we can use:
    inv2 = (mod + 1) // 2  # alternatively, pow(2, mod - 2, mod)
    ans = (num * inv2) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()