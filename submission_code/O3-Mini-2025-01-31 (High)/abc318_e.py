def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    A = list(map(int, data[1:]))

    # Group indices for each value.
    # We use 1-indexing as given in the problem.
    groups = {}
    for i, a in enumerate(A, 1):
        if a not in groups:
            groups[a] = []
        groups[a].append(i)

    ans = 0
    # For each group corresponding to a value x:
    # Let the sorted list of indices be L = [L0, L1, ..., L(m-1)]
    # For any pair (L[p], L[q]) with p < q, the number of valid j such that 
    # L[p] < j < L[q] and A[j] != x equals (L[q] - L[p] - 1) minus the number 
    # of occurrences of x between them (which is q - p - 1). So the contribution is:
    # (L[q] - L[p] - 1) - (q - p - 1) = L[q] - L[p] - (q - p)
    # This can be reformulated as: (L[q] - q) - (L[p] - p)
    # To efficiently sum over all pairs, we define for each occurrence, diff = L[i] - i (with i 0-indexed).
    # Then total contribution for this group is: sum_{i < j} (diff[j] - diff[i]).
    
    for positions in groups.values():
        m = len(positions)
        if m < 2:
            continue
        prefix_sum = 0
        # We'll compute the sum efficiently by iterating over the list:
        for i in range(m):
            diff = positions[i] - i  # positions use 1-index, 'i' is 0-index in the group.
            ans += diff * i - prefix_sum
            prefix_sum += diff

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()