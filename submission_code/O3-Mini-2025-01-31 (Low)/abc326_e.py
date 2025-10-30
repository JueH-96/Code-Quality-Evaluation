def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    mod = 998244353
    it = iter(input_data)
    N = int(next(it))
    # The array is 1-indexed.
    A = [0]*(N+1)
    for i in range(1, N+1):
        A[i] = int(next(it)) % mod

    # Note on the process:
    # Let F(x) be the expected additional salary when current highest roll is x.
    # When x is between 0 and N, on a new roll (uniformly among 1..N):
    # - If y > x, we get A[y] yen and state updates to y (thus adding F(y))
    # - If y <= x, the process stops.
    #
    # In particular, for state x (with 0 <= x < N):
    #   F(x) = (1/N)*sum_{y=x+1}^{N} (A[y] + F(y))
    # for x = N we have F(N) = 0.
    #
    # Let S(i) = A[i] + F(i) + A[i+1] + F(i+1) + ... + A[N] + F(N) for i in {1,...,N+1},
    # with S(N+1) = 0.
    # Notice for i in 1..N, we have:
    #   F(i) = (1/N)*S(i+1)
    # and so,
    #   S(i) = A[i] + F(i) + S(i+1)
    #        = A[i] + (1/N)*S(i+1) + S(i+1)
    #        = A[i] + ((N+1)/N)*S(i+1).
    #
    # We need F(0) and note that from state 0:
    #   F(0) = (1/N)*(A[1]+F(1) + A[2]+F(2) + ... + A[N]+F(N))
    #        = (1/N)*S(1)
    #
    # This recurrence expresses S(1) in terms of A[i] multiplied by powers of (N+1)/N.
    # We get:
    #   S(1) = A[1] 
    #        + ((N+1)/N)*A[2] 
    #        + ((N+1)/N)^2*A[3] 
    #        + ... 
    #        + ((N+1)/N)^(N-1)*A[N].
    #
    # So the answer (expected salary) is F(0)=S(1)/N.
    #
    # Compute c = (N+1)/N mod mod and then S(1) = sum_{i=1}^N A[i]*(c^(i-1)).
    
    invN = pow(N, mod-2, mod)
    c = ((N+1) % mod) * invN % mod
    
    cs = 1
    total = 0
    for i in range(1, N+1):
        total = (total + A[i] * cs) % mod
        cs = cs * c % mod
    
    ans = total * invN % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()