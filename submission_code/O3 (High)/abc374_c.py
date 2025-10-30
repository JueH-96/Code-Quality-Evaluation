import sys
import bisect


def subset_sums(arr):
    """Return list of all subset sums of ``arr`` ( |arr| ≤ 10 )."""
    sums = [0]
    for v in arr:
        # add v to every existing element to create new subset sums
        sums += [x + v for x in sums]
    return sums


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    n = data[0]
    k = data[1:]

    total = sum(k)
    best_possible = (total + 1) // 2          # ceil(total / 2)

    # ----- meet-in-the-middle: split the array into two halves
    mid = n // 2
    left, right = k[:mid], k[mid:]

    left_sums = subset_sums(left)             # at most 2^10 = 1024 elements
    right_sums = subset_sums(right)
    right_sums.sort()

    best = total                              # worst case (all in one group)

    # search for subset sum closest to total/2
    for s_left in left_sums:
        target = total // 2 - s_left
        idx = bisect.bisect_left(right_sums, target)

        # examine the closest candidates around idx
        for j in (idx - 1, idx):
            if 0 <= j < len(right_sums):
                s = s_left + right_sums[j]
                best = min(best, max(s, total - s))
                if best == best_possible:     # cannot improve further
                    print(best)
                    return

    print(best)


if __name__ == "__main__":
    main()