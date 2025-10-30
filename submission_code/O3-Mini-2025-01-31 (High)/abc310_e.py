def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    s = data[1].strip()

    # Explanation:
    # We want to sum, over all subarrays f(i,j) where
    #    f(i,i) = A_i   and for j > i:  f(i,j) = f(i,j-1) NAND A_j.
    #
    # Recall: NAND is defined as:
    #   0 NAND 0 = 1,  0 NAND 1 = 1,  1 NAND 0 = 1,  1 NAND 1 = 0.
    #
    # A key observation is that for any subarray of length ≥ 2,
    #   f(i,j) = 0   if and only if (f(i,j-1) == 1 and A_j == 1);
    #   otherwise f(i,j) = 1.
    #
    # For a fixed endpoint j (1-indexed), let P(j) be the number of subarrays ending at j
    # (i.e. with starting index i, 1 <= i <= j) that yield f(i,j) = 1.
    #
    # Notice that:
    # For a subarray of length 1, f(i,i) = A_i, so P(j) gets an extra 1 if A_j is '1' and 0 if '0'.
    #
    # For subarrays of length ≥ 2, when extending a subarray that ended at j-1:
    #   • If extending with A_j == '0': then regardless of f(i,j-1), we get f(i,j) = NAND(_,0) = 1.
    #     Thus from all subarrays ending at j-1 (there are j-1 of them), all extended versions yield 1.
    #     And the new subarray (j,j) contributes 0 because f(j,j) = A_j ('0' gives 0).
    #     So overall, when A_j == '0', we have P(j) = (j-1) + 0 = j-1.
    #
    #   • If extending with A_j == '1': then for each subarray ending at j-1,
    #     we have f(i,j) = 0 if f(i,j-1) was 1, and f(i,j) = 1 if f(i,j-1) was 0.
    #     In other words, the number of extended subarrays yielding 1 is:
    #         (number of subarrays ending at j-1 with f = 0)
    #         = (j-1) - P(j-1).
    #     And the new subarray (j,j) gives f(j,j) = 1.
    #     So in total, for A_j == '1': P(j) = (j-1 - P(j-1)) + 1 = j - P(j-1).
    #
    # Thus our recurrence for j>=2 (1-indexed) is:
    #     if A_j == '0':  P(j) = j - 1.
    #     if A_j == '1':  P(j) = j - P(j-1).
    # For j = 1:  P(1) is int(A_1) (i.e. 1 if A_1=='1', otherwise 0).
    #
    # The answer is the total sum of f(i,j) over all subarrays.
    # Since for subarrays of length ≥ 2, the value is 1 exactly when f(i,j)==1,
    # and for subarrays of length 1 the contribution is exactly A_i, then
    # the answer is simply:
    #      ans = sum_{j=1}^{n} P(j)
    #
    # We now compute P(1), P(2), ..., P(n) in one pass.
    
    ans = 0
    # For index 0 (which is j=1 in 1-indexing)
    p = 1 if s[0] == '1' else 0
    ans += p
    # Process j from 1 to n-1 (corresponding to positions 2 to n in 1-indexing)
    for j in range(1, n):
        pos = j + 1  # current endpoint in 1-indexed terms
        if s[j] == '0':
            p = pos - 1  # when the new bit is '0', all previous subarrays extend to 1
        else:
            p = pos - p  # when the new bit is '1', the ones that were 0 become 1, plus new subarray contributes 1
        ans += p
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()