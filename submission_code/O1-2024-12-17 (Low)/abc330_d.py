def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1:]

    # For each column, count how many 'o' in that column.
    # col_count[c] = number of rows that have 'o' in column c.
    col_count = [0] * N
    # For each row, we'll keep a list of column-indices where S[r][c] == 'o'.
    row_cols = [[] for _ in range(N)]

    for r in range(N):
        for c in range(N):
            if S[r][c] == 'o':
                row_cols[r].append(c)
                col_count[c] += 1

    # Now we compute the answer.
    # The logic:
    # For each row r, let its columns-with-'o' be colList = row_cols[r].
    # We want to count the number of pairs (i < j) in colList,
    # and then the third cell can be chosen by sharing the column with either i or j.
    # The contribution for each pair (colList[i], colList[j]) is:
    #       (col_count[colList[i]] - 1) + (col_count[colList[j]] - 1)
    # = col_count[colList[i]] + col_count[colList[j]] - 2
    #
    # Summing over all pairs can be done in O(k) after sorting + prefix sums, where k is
    # number of 'o' in that row.
    #
    # We'll implement that carefully to avoid O(k^2) loops.

    ans = 0

    for r in range(N):
        col_list = row_cols[r]
        k = len(col_list)
        if k < 2:
            continue
        # Sort col_list (not strictly necessary to do ascending, but helpful for prefix sums)
        col_list.sort()

        # Build prefix sums of col_count for these columns
        prefix = [0] * (k + 1)
        for i in range(k):
            prefix[i+1] = prefix[i] + col_count[col_list[i]]

        # A = sum_{0 <= i < j < k} col_count[col_list[i]]
        #   = sum_{j=1..k-1} sum_{i=0..j-1} col_count[col_list[i]]
        #   = sum_{j=1..k-1} prefix[j]
        # We just need to sum prefix[j] for j=1..(k-1).

        # B = sum_{0 <= i < j < k} col_count[col_list[j]]
        #   = sum_{j=1..k-1} col_count[col_list[j]] * j

        A = 0
        for j in range(1, k):
            A += prefix[j]

        B = 0
        for j in range(1, k):
            B += col_count[col_list[j]] * j

        # Then total for row r is:
        # sum_{pairs} (col_count[c1] + col_count[c2]) - 2*(#pairs)
        # #pairs = k*(k-1)/2
        # so subtract 2 * (k*(k-1)/2) = k*(k-1)

        # sumPairs = A + B - k*(k-1)

        sum_pairs = A + B - k*(k-1)
        ans += sum_pairs

    print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()