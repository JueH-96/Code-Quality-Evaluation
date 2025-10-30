def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    mod = 998244353

    N = int(input_data[0])
    # Read A[1]...A[N]; use 1-indexing.
    A = [0] * (N + 1)
    for i in range(N):
        A[i + 1] = int(input_data[i + 1]) % mod

    # The process starts with x=0. We define E(m) as the expected additional salary
    # when the current maximum is m.
    # At state m (0 <= m < N), the next roll has:
    #   • probability m/N to yield a number ≤ m (termination)
    #   • probability (N-m)/N to yield something > m.
    # If the roll is an improvement (i.e. a value i with m < i ≤ N),
    # then with probability 1/(N-m) the specific outcome is i, resulting in an immediate payment A[i] and 
    # a change of state to i (with expected additional earning E(i)).
    # Therefore, for 0 <= m < N, we have:
    #    E(m) = (1/N)*sum_{i=m+1}^{N} (A[i] + E(i)).
    # And for m = N, E(N) = 0.
    #
    # In our problem we are asked for E(0). Notice that:
    #    E(0) = (1/N)*sum_{i=1}^{N} (A[i] + E(i)).
    # If we define T(m) = sum_{i=m}^{N} (A[i] + E(i)) for m>=1, then:
    #    E(m) = T(m+1)/N  (for 0 <= m < N)
    # and in particular,
    #    T(m) = A[m] + E(m) + T(m+1) = A[m] + T(m+1) + (T(m+1)/N).
    # That is, for 1 <= m < N:
    #    T(m) = A[m] + ((N+1)/N) * T(m+1)
    # with the boundary T(N) = A[N] (since E(N)=0).
    #
    # Finally, E(0) = (1/N) * T(1)  (since the first roll always counts as an improvement).
    # We now compute T[i] for i = N, N-1, ... , 1, and then answer = T(1)/N modulo mod.

    # Precompute the modular inverse of N
    invN = pow(N, mod-2, mod)
    # Factor representing (N+1)/N mod mod.
    factor = ((N+1) % mod) * invN % mod

    # Compute T[1]..T[N] in descending order.
    # We'll use 1-indexing. Let T[N+1] be 0.
    T = [0] * (N + 2)
    T[N] = A[N] % mod
    for i in range(N - 1, 0, -1):
        T[i] = (A[i] + factor * T[i + 1]) % mod

    # The expected total salary is given by E(0) = T(1)/N mod mod.
    ans = T[1] * invN % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()