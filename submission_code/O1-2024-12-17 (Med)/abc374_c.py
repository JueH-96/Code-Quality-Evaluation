def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = list(map(int, input_data[1:]))

    # Split departments into two halves
    half = N // 2
    first_half = K[:half]
    second_half = K[half:]

    # Function to get all possible subset sums of a given list
    def get_all_sums(arr):
        sums = [0]
        for val in arr:
            new_sums = []
            for s in sums:
                new_sums.append(s + val)
            sums += new_sums
        return sums

    sums_first = get_all_sums(first_half)
    sums_second = get_all_sums(second_half)
    sums_second.sort()  # sort for binary search

    total_sum = sum(K)
    best = total_sum  # a trivial upper bound (all in one group)

    # For each subset sum in the first half, find the best match in the second half
    for s1 in sums_first:
        # We want the total in Group A (s1 + s2) to be as close to total_sum//2 as possible
        x = (total_sum // 2) - s1
        # Find the insertion point in sums_second
        pos = bisect.bisect_left(sums_second, x)

        # Check neighbors around pos
        for check_pos in [pos - 1, pos, pos + 1]:
            if 0 <= check_pos < len(sums_second):
                s2 = sums_second[check_pos]
                group_a = s1 + s2
                group_b = total_sum - group_a
                best = min(best, max(group_a, group_b))

    print(best)

# Do not forget to call main()
if __name__ == "__main__":
    main()