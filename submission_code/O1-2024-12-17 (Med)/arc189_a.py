def main():
    import sys
    MOD = 998244353

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # --------------------------------------------------------
    # 1) The color of cell 1 is initially 1 (since 1 mod 2 = 1)
    #    and it can never change (there is no way to enclose it),
    #    so A[0] must be 1 for there to be any valid sequence.
    #
    # 2) The color of cell N is initially (N mod 2) and it can
    #    never change either, so A[N-1] must be (N mod 2).
    #
    # 3) For 1-based indexing, all odd cells start as 1 and
    #    cannot become 0 unless enclosed by two 0 endpoints.
    #    But the leftmost boundary (cell 1) is forced to stay 1
    #    and cannot ever become 0, so there is no way to create
    #    a 0 endpoint on the far left.  In fact, in such a chain,
    #    one cannot flip any odd cell to 0 at all — there just
    #    aren't two zero endpoints to enclose an odd cell.
    #
    #    Conclusion: all odd cells must remain 1 in the final
    #    configuration if it is to be reachable.
    #
    # 4) Even cells start as 0, but they can become 1 if enclosed
    #    by two 1 endpoints (which come from odd cells that remain
    #    1, including possibly the first cell or some internal odd
    #    cell).  We can flip some subset (possibly empty) of the
    #    even cells from 0 to 1.
    #
    # 5) It turns out (and can be shown) that if you have k even
    #    cells that end as 1, the number of distinct operation
    #    sequences that achieve this (under the problem's rules)
    #    is 3^(k-1)  (with the convention that 3^(-1) = 1 if k=0).
    #
    #    This pattern is illustrated by the samples:
    #      - Sample 1 (k=2) => 3 ways
    #      - Sample 2 (k=3) => 9 ways (3^2)
    #      - If k=1 => exactly 1 way (3^0 = 1)
    #      - If k=0 => do nothing => 1 way
    #
    # Steps to solve:
    #
    #  a) Check A[0] == 1, and A[N-1] == N%2; otherwise answer=0.
    #  b) Check that all odd indices (1-based) in A are still 1.
    #     That means A[j] == 1 for j=0,2,4,... in 0-based indexing.
    #  c) Count how many even indices (1-based) have become 1
    #     in A.  In 0-based terms, these are j=1,3,5,... that
    #     are 1.  Let that count be k.
    #  d) If k==0, answer = 1, else answer = 3^(k-1) mod 998244353.
    #
    # --------------------------------------------------------

    # 1) Boundary checks
    if A[0] != 1:  # cell 1 must remain 1
        print(0)
        return

    if A[-1] != (N % 2):  # cell N must remain N mod 2
        print(0)
        return

    # 2) All odd-indexed cells (in 1-based sense) must remain 1
    #    i = 1,3,5,... => 0-based indices 0,2,4,...
    for j in range(0, N, 2):
        if A[j] != 1:
            print(0)
            return

    # 3) Count how many even-indexed cells (1-based) ended up as 1
    #    i = 2,4,6,... => 0-based indices 1,3,5,...
    k = 0
    for j in range(1, N, 2):
        if A[j] == 1:
            k += 1

    # 4) Number of ways = 3^(k-1) if k>0, else 1
    if k == 0:
        print(1)
    else:
        # Compute 3^(k-1) mod 998244353
        # (Python's built-in pow(base, exp, mod) does fast exponentiation)
        print(pow(3, k-1, MOD))

# Do not forget to call main() at the end!
if __name__ == "__main__":
    main()