def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Parse inputs
    N = int(data[0])
    M = int(data[1])
    # Next M numbers are the cell positions and then M numbers are the number of stones.
    xs = list(map(int, data[2:2 + M]))
    As = list(map(int, data[2 + M:2 + 2 * M]))
    
    # The total number of stones must equal N (since in the final state each of the N cells must have 1 stone).
    if sum(As) != N:
        print(-1)
        return

    # Prepare a sorted list of the stone-holding cells.
    stones_list = list(zip(xs, As))
    stones_list.sort(key=lambda x: x[0])

    # The transformation we want can be rephrased in terms of adjusting the differences.
    # Let d[i] = (stones in cell i) - 1. We want all cells to have 0 difference.
    # Only moves allowed are from cell i to cell i+1.
    # With this restriction, a necessary (and in fact sufficient) condition is that
    # the cumulative sums (prefix sums) P(i) = sum_{j=1}^{i} d[j] are all nonnegative (for i=1,...,N-1)
    # because we cannot “pull” stones backward.
    #
    # A known fact about adjacent transfer problems is that the minimal number of operations required
    # is given by the sum of the prefix imbalances:
    #       answer = sum_{i=1}^{N-1} P(i)
    # where P(i) = (sum of stones in cells 1..i) - i.
    #
    # In our problem almost all cells initially contain 0 stones, except for the M given cells.
    # Hence, P(i) remains constant on intervals where no stone is added.
    # We can “integrate” over these intervals using arithmetic series formulae.

    # Helper: given a constant value c over cells from L to R, compute
    # gap_sum = sum_{i=L}^{R} (c - i).
    def gap_sum(L, R, c):
        count = R - L + 1
        # Sum of integers from L to R is (R*(R+1)//2 - (L-1)*L//2)
        total_i = (R * (R + 1) // 2) - ((L - 1) * L // 2)
        return count * c - total_i

    ans = 0
    current_prefix = 0  # This will hold (sum of stones processed so far) - (number of cells processed)
    last = 0  # Last cell index we have processed

    # Process each stone-holding cell in increasing order.
    for pos, a in stones_list:
        # Process the gap between the last processed cell and this stone cell.
        # In all these cells there is no change in stone count so the prefix is constant.
        if last + 1 <= pos - 1:
            L = last + 1
            R = pos - 1
            # For feasibility, we must have the prefix for every cell in [L,R] nonnegative.
            # Because the prefix here is constant = current_prefix, it is enough to check that
            # current_prefix >= R (the worst-case, since i is largest at R).
            if current_prefix < R:
                print(-1)
                return
            ans += gap_sum(L, R, current_prefix)
        # Now process the cell at index pos.
        new_prefix = current_prefix + a  # this is P(pos) = (previous stones) + a
        if new_prefix < pos:
            # Feasibility: We need new_prefix - pos >= 0; that is, P(pos) must be nonnegative.
            print(-1)
            return
        ans += (new_prefix - pos)
        current_prefix = new_prefix
        last = pos

    # Process the final gap from the last stone cell to cell N - 1.
    if last < N:
        L = last + 1
        R = N - 1
        if current_prefix < R:
            print(-1)
            return
        ans += gap_sum(L, R, current_prefix)

    print(ans)

if __name__ == '__main__':
    main()