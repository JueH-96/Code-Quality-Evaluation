import sys
import threading

def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A = int(data[0]); B = int(data[1]); M = int(data[2])
    # We count permutations of size N = A*B - 1 whose RSK-shape is
    # lambda = (A,A,...,A, A-1) with B-1 rows of length A and one row of length A-1,
    # and which satisfy the "n+0.5" condition.  One shows that the total count is
    # f^lambda * g, where f^lambda is the number of SYT of shape lambda
    # given by the hook-length formula, and
    #    g = C(A+B-3, A-2).
    #
    # We work modulo the prime M.
    N = A * B - 1
    # We need factorials up to max(N, A+B-3).
    LIM = max(N, A + B - 3)
    # Precompute factorials and inverses mod M
    fact = [1] * (LIM + 1)
    for i in range(1, LIM + 1):
        fact[i] = fact[i-1] * i % M
    invfact = [1] * (LIM + 1)
    # Fermat inverse of fact[LIM]
    invfact[LIM] = pow(fact[LIM], M-2, M)
    for i in range(LIM, 0, -1):
        invfact[i-1] = invfact[i] * i % M

    # Compute the hook‐length product for lambda = (A^ (B-1), A-1)
    # hooks at positions (i,j):
    #   if 1 <= i <= B-1 and 1 <= j <= A-1:
    #       h = A + B - i - j + 1
    #   if 1 <= i <= B-1 and j = A:
    #       h = B - i
    #   if i = B and 1 <= j <= A-1:
    #       h = A - j
    hook_prod = 1
    # rows 1..B-1
    for i in range(1, B):
        # columns 1..A-1
        # hook(i,j) = A + B - i - j + 1
        # we can loop j
        base = A + B - i + 1
        for j in range(1, A):
            h = base - j
            hook_prod = (hook_prod * h) % M
        # column j = A
        h = B - i
        hook_prod = (hook_prod * h) % M
    # row i = B, columns 1..A-1
    for j in range(1, A):
        h = A - j
        hook_prod = (hook_prod * h) % M

    # f^lambda = N! / (product of hooks)  mod M
    f_lambda = fact[N] * pow(hook_prod, M-2, M) % M

    # g = C(A+B-3, A-2)
    s = A + B - 3
    k = A - 2
    if 0 <= k <= s:
        g = fact[s] * invfact[k] % M * invfact[s-k] % M
    else:
        g = 0

    ans = f_lambda * g % M
    print(ans)

if __name__ == "__main__":
    main()