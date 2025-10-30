def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0].strip()
    n = len(s)
    if n < 3:
        sys.stdout.write("0")
        return

    from collections import defaultdict
    # Group indices for each letter. We use 0-index positions.
    letter_positions = defaultdict(list)
    for idx, ch in enumerate(s):
        letter_positions[ch].append(idx)

    total = 0
    # For any triple (i, j, k) with i < j < k to form a palindrome,
    # we need S[i] == S[k]. For each letter, if its positions are:
    # L[0], L[1], ... L[m-1] then for each pair (L[i], L[j])
    # with i < j the number of valid j (middle index in triple) is:
    # (L[j] - L[i] - 1)
    # So the total count for that letter is:
    # Sum_{0 <= i < j < m} (L[j] - L[i] - 1)
    # We can compute this quickly using a prefix sum.
    for lst in letter_positions.values():
        m = len(lst)
        if m < 2:
            continue
        prefix = [0] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] + lst[i]
        # For each starting position L[i], sum contributions for all later positions:
        for i in range(m - 1):
            count = m - i - 1  # Number of valid pairs with fixed i
            sum_after = prefix[m] - prefix[i + 1]  # Sum of L[j] for j > i
            # For fixed i, contribution: sum_{j=i+1}^{m-1} (L[j] - L[i] - 1)
            total += (sum_after - count * (lst[i] + 1))
    
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()