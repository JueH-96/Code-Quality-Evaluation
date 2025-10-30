def main():
    import sys
    mod = 998244353

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    
    # Special case: if there is only one ball, the answer is trivially 1.
    if N == 1:
        sys.stdout.write("1")
        return

    # We derive the following recurrence for the expected position.
    # Let E be the expected position of the black ball and suppose it is in position i.
    # In one swap operation, two distinct positions a and b are chosen uniformly from 1...N.
    # The probability that the current black ball at position i gets involved in the swap is:
    #    P(moving) = 2*(N-1)/(N*(N-1)) = 2/N.
    # In that swap, it moves to the other position (call it j). Since all j (j != i) are equally likely,
    # its new expected position is the average of all positions except i.
    #
    # That is, given that sum of all positions is S = N(N+1)/2, the average among j ≠ i is:
    #    (S - i)/(N-1).
    #
    # Therefore, the expected new position given current position i is:
    #    E' = (1 - 2/N)* i  +  (2/N) * [ (S - i)/(N-1) ]
    # Plug S = N(N+1)/2:
    #    E' = (1 - 2/N)* i + (2/N)*[ (N(N+1)/2 - i)/(N-1) ]
    #       = i - (2i)/N +  (N+1)/(N-1) - (2*i)/(N*(N-1))
    #       = i * (1 - 2/N - 2/(N(N-1)))  +  (N+1)/(N-1)
    # Notice that:
    #    1 - (2/N) - (2/(N(N-1))) = 1 - [2(N-1) + 2] / (N(N-1)) = 1 - (2N)/(N(N-1)) = 1 - (2/(N-1))
    #
    # So we have a linear recurrence:
    #    E' = A*i + B
    # with A = 1 - 2/(N-1)  and  B = (N+1)/(N-1)
    #
    # Starting with initial position E0 = 1, after K independent steps we have:
    #    E(K) = A^K * 1 + B*(1 + A + A^2 + ... + A^(K-1))
    # And the geometric series sum equals (1 - A^K) / (1 - A) where 1-A = 2/(N-1):
    #    E(K) = A^K + B * ((1-A^K)/(2/(N-1)))
    #         = A^K + B*(N-1)*(1-A^K)/2
    # Plugging B back in:
    #         = A^K + ((N+1)/(N-1))*(N-1)*(1-A^K)/2
    #         = A^K + (N+1)*(1-A^K)/2
    #         = (N+1)/2 - (N+1 - 2*A^K)/2? Let's rewrite cleanly.
    #
    # Alternatively, re-arrange as:
    #    E(K) = ((N+1) - (N-1)*A^K) / 2.
    #
    # Here, A = 1 - 2/(N-1) = (N-3)/(N-1).
    #
    # In modulo arithmetic we need to compute the fraction and the modular inverse of denominators.
    # The answer should be reported as the unique integer R with 0 <= R < mod satisfying
    #     R * Q  ≡ P  (mod mod)
    # where the expected value is represented as P/Q in simplest form.
    
    # In the following, we compute:
    #    ans =  ((N+1) - (N-1) * ((N-3)/(N-1))^K )/2   mod mod

    # Define a small function for modular inversion.
    inv = lambda a: pow(a, mod - 2, mod)

    # Compute (N+1) mod mod and (N-1) mod mod.
    num1 = (N + 1) % mod
    num2 = (N - 1) % mod

    # Compute A = (N-3)/(N-1) mod mod.
    A_num = (N - 3) % mod
    A_den = (N - 1) % mod
    A = (A_num * inv(A_den)) % mod

    # Compute A^K mod mod.
    Aexp = pow(A, K, mod)

    # Our expected value is given by: ans = (num1 - num2 * Aexp) / 2 mod mod.
    ans = (num1 - num2 * Aexp) % mod
    ans = ans * inv(2) % mod

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()