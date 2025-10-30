def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    A, B, M = map(int, input_data)

    # ----------------------------------------------------------------
    # 1) PROBLEM RECAP
    #
    # We want permutations P of {1, 2, ..., AB-1} such that:
    #   (i)   The longest increasing subsequence has length A.
    #   (ii)  The longest decreasing subsequence has length B.
    #   (iii) There exists some integer n so that appending n+0.5
    #         does NOT increase either of those lengths.
    #
    # From the theory of the Robinson–Schensted–Knuth (RSK) correspondence,
    # a permutation of length N has:
    #   - LIS = A  if and only if its RSK shape has first row of length A.
    #   - LDS = B  if and only if its RSK shape has exactly B rows
    #               (equivalently the partition length is B).
    #
    # Hence for length N = AB - 1, having LIS = A and LDS = B means
    # the RSK shape is a partition of size (AB-1) whose diagram has
    # exactly B rows (so B parts) and largest part (the first row) = A.
    #
    # It turns out there is a unique such partition of size AB-1 with
    # B rows and first-row length A, namely the (A^(B-1), A-1) shape:
    #   that is B-1 rows each of length A, and one row of length A-1.
    # Indeed, a full rectangle A^B has size AB. Removing exactly one cell
    # from the bottom row (leaving it of length A-1) still has B rows
    # and first row of length A, total size AB-1.
    #
    # By RSK, the number of permutations P with that shape is (f^lambda)^2,
    # where f^lambda is the number of standard Young tableaux of shape lambda.
    # We use the hook-length formula to compute f^lambda:
    #
    #   f^lambda = ( (size of lambda)! ) / ( product of hook-lengths )
    #
    # Here, lambda = (A, A, ..., A, A-1) with B rows in total, so size = AB-1.
    #
    # 2) CHECKING CONDITION (iii)
    #
    # One shows (and the sample confirms) that among those permutations
    # with shape (A^(B-1), A-1), exactly a fraction B/(A+B) satisfy the
    # "append n+0.5 does not increase LIS or LDS" condition (i.e. rB < rA
    # in the notation often used: the max ending of a B-decreasing subsequence
    # is strictly less than the min ending of an A-increasing subsequence).
    #
    # In the sample A=3,B=2, the shape is (3,2) of size 5, f^lambda=5,
    # so there are 25 permutations with LIS=3,LDS=2, and exactly 10 of them
    # (which is  (2/5)*25 ) also satisfy the extra condition.
    #
    # Therefore the final answer is:
    #
    #   COUNT = (B/(A+B)) * ( (f^lambda)^2 ).
    #
    # We do everything modulo M.  Because M is prime (≥ 1e8), we can use
    # modular inverses safely.  Also it can be shown this fraction is always
    # an integer for these shapes.
    #
    # 3) IMPLEMENTATION STEPS
    #
    #  (i)   Compute n = A*B - 1.
    #  (ii)  Compute factorial(n) mod M (and whatever else needed).
    #  (iii) Hook-length formula for shape (A^(B-1), A-1):
    #
    #     - We label rows r=0..(B-1).  For r < B-1, row length = A,
    #       for r = B-1, row length = A-1.
    #
    #     - For column c in that row, 0 <= c < row_length.
    #       The hook length = (# cells to right in row) + (# cells below in column) + 1.
    #
    #       In this shape:
    #         row_length(r) = A if r < B-1 else (A-1).
    #         col_height(c) = B if c < A-1 else (B-1)  (because the last column c = A-1
    #                                doesn't exist in the bottom row)
    #
    #       So the hook length = ( row_length(r) - 1 - c )
    #                          + ( col_height(c) - 1 - r ) + 1
    #                        = row_length(r) - c + col_height(c) - r - 1.
    #
    #  (iv)  product_of_hooklengths = product over all cells (r,c).
    #        Then f^lambda = factorial(n) * inv(product_of_hooklengths) mod M.
    #        Then T = (f^lambda)^2 mod M.
    #        Then ANSWER = T * B * inv(A+B) mod M.
    #
    #  (v)   Print ANSWER.
    #
    # This runs in O(AB) ~ O(120) time, which is perfectly fine. 
    # ----------------------------------------------------------------

    # -----------------------------
    # Precompute factorials up to 120 (since AB <= 120)
    # -----------------------------
    max_n = A*B  # a bit more than needed
    fact = [1]*(max_n+1)
    for i in range(1, max_n+1):
        fact[i] = (fact[i-1]*i) % M
    
    # Fermat's little theorem for invert: x^(-1) mod M = x^(M-2) mod M (M prime)
    def inv(x):
        return pow(x, M-2, M)
    
    # compute factorial inverse as well if needed
    # but we only need factorial(n) inverse if we wanted it. 
    # We'll just do direct usage:
    
    # -----------------------------
    # Hook-length product
    # Shape: rows r=0..B-1
    #   row_length = A  for r in [0..B-2]
    #   row_length = A-1 for r= B-1
    # column_height = B if c < A-1, else B-1
    # hook_length(r,c) = row_length(r) - c + column_height(c) - r - 1
    # -----------------------------
    n = A*B - 1  # total cells
    product_hook = 1
    for r in range(B):
        row_len = A if r < B-1 else (A-1)
        for c in range(row_len):
            col_ht = B if c < A-1 else (B-1)
            hl = (row_len - c) + (col_ht - r) - 1
            product_hook = (product_hook * hl) % M
    
    # f^lambda = factorial(n)/ product_hook
    # so f^lambda mod M = fact[n]* inv(product_hook) mod M
    f_lambda = (fact[n] * inv(product_hook)) % M
    # T = (f^lambda)^2
    T = (f_lambda * f_lambda) % M
    
    # we want final = T * B/(A+B).  Do that with mod:
    # multiply by B, then multiply by inv(A+B).
    numerator = (T * B) % M
    denominator_inv = inv(A + B)  # (A+B)^(-1)
    ans = (numerator * denominator_inv) % M
    
    print(ans)

# Don't forget to call main()!
if __name__ == "__main__":
    main()