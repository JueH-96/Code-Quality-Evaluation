def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    # Although the sizes are given as sorted, we enforce it.
    A.sort()

    # Explanation:
    # We want to form as many pairs as possible such that for a pair (a, b),
    # where a is the top (smaller) mochi and b is the bottom (larger) mochi,
    # the condition 2 * a <= b holds.
    # We use a two-pointers greedy method:
    # 1. Use the first half of A (the smaller elements) as candidates for the top.
    # 2. Use the second half for the base.
    # For each candidate from the left pointer (i) and from the right pointer (j),
    # if the condition is met, we form a pair and move both pointers.
    # Otherwise, we move the right pointer (j) to try with a larger mochi.
    # This approach ensures that each mochi is used at most once.

    left = 0
    # For an even number of mochi, using n//2 works.
    # For an odd number, it is better to start j from n//2 since the number
    # of possible pairs is at most n//2.
    right = n // 2
    count = 0

    while left < n // 2 and right < n:
        # If the candidate at left can be paired with candidate at right,
        # i.e. 2*A[left] <= A[right], then record the pair.
        if 2 * A[left] <= A[right]:
            count += 1
            left += 1
            right += 1
        else:
            right += 1

    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()