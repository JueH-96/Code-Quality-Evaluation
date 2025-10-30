def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    pos_in = 1
    # Explanation:
    # We are given 2N positions where each number 1..N appears exactly twice.
    # For a couple a with positions (i, j) (i < j) and a couple b with positions (k, l) (k < l)
    # (and neither couple’s occurrences are adjacent) it turns out that one swap can put
    # both couples together if and only if one can choose a swap so that after the swap
    # the new occurrences become adjacent.
    #
    # A case analysis shows that, when one swaps one occurrence of a with one occurrence of b,
    # there are four possible ways. Only one possibility actually works (if any) – it requires:
    #
    #    (i) For couple a: positions (i, j) and for couple b: positions (k, l)
    #         with i < j and k < l, one possible successful swap is swapping a at position i
    #         with b at position l.
    #    (ii) After the swap, a appears at positions j and l while b appears at i and k.
    #         For a to become adjacent, we need either |j - l| = 1; for b to be adjacent, |i - k| = 1.
    #
    # With the natural order (since i < j and k < l) the only possibility is:
    #         l = j + 1 and k = i + 1.
    #
    # Alternatively, one could use the swap A[j] with A[k]. Then a would occur at i and k (adjacent if k = i+1)
    # and b would occur at j and l (adjacent if l = j+1). In either case the necessary and sufficient condition is:
    #
    #     For couples a and b having positions (i, j) and (k, l) respectively (with i < j and k < l),
    #     a one‐swap solution exists if and only if
    #
    #         k = i + 1   and   l = j + 1.
    #
    # In other words, if we list the couples in increasing order of their first occurrence,
    # two couples that can be “fixed” by a single swap must have their first-occurrence positions consecutive
    # and also their second-occurrence positions consecutive.
    #
    # Also note that the problem statement excludes couples that are initially next to each other.
    #
    # So the solution is:
    #   1. For each couple, record its two positions.
    #   2. Skip the couple if the two positions are adjacent.
    #   3. Sort the remaining couples by their first occurrence.
    #   4. Count consecutive couples in this sorted order that satisfy:
    #         current.first = previous.first + 1  and  current.second = previous.second + 1.
    
    out_lines = []
    for _ in range(t):
        n = int(data[pos_in]); pos_in += 1
        arr = list(map(int, data[pos_in:pos_in + 2 * n]))
        pos_in += 2 * n
        
        # Record the two positions for each couple (1-indexed)
        # pos[a] will be a list [first_occurrence, second_occurrence]
        pos_arr = [[None, None] for _ in range(n + 1)]
        for idx, a in enumerate(arr, start=1):
            if pos_arr[a][0] is None:
                pos_arr[a][0] = idx
            else:
                pos_arr[a][1] = idx
        
        valid_couples = []
        for a in range(1, n + 1):
            i, j = pos_arr[a]
            if j - i == 1:
                # the couple is originally adjacent so ignore
                continue
            valid_couples.append((i, j))
        
        # Sort couples by first occurrence
        valid_couples.sort(key=lambda x: x[0])
        
        count = 0
        # Only consecutive couples in the sorted order might satisfy:
        # let previous couple: (i, j) and current couple: (k, l)
        # then we need k = i + 1 and l = j + 1.
        for idx in range(1, len(valid_couples)):
            prev_i, prev_j = valid_couples[idx-1]
            curr_i, curr_j = valid_couples[idx]
            if curr_i == prev_i + 1 and curr_j == prev_j + 1:
                count += 1
        
        out_lines.append(str(count))
    
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()