import sys


def main() -> None:
    # read everything at once (fast and simple)
    data = list(map(int, sys.stdin.buffer.read().split()))
    N, Q = data[0], data[1]
    xs = data[2: 2 + Q]          # queries (exactly Q numbers)

    # insertion_idx[i] = query index at which i was inserted, 0 if currently not in S
    insertion_idx = [0] * (N + 1)

    # answer array (1-based for convenience)
    A = [0] * (N + 1)

    # prefix sums of |S| after every query
    # prefix[k] = s1 + s2 + ... + sk     (prefix[0] = 0)
    prefix = [0] * (Q + 1)

    current_size = 0  # |S| right after processing the current query

    for i in range(1, Q + 1):
        x = xs[i - 1]

        if insertion_idx[x]:                       # x is currently in S  ->  remove
            start = insertion_idx[x]               # when it was inserted
            # x received |S| for queries start .. i-1
            A[x] += prefix[i - 1] - prefix[start - 1]
            insertion_idx[x] = 0
            current_size -= 1
        else:                                      # x not in S  ->  insert
            current_size += 1
            insertion_idx[x] = i

        # store prefix sum up to this query
        prefix[i] = prefix[i - 1] + current_size

    total_prefix = prefix[Q]

    # close all intervals that are still active after the last query
    for x in range(1, N + 1):
        start = insertion_idx[x]
        if start:
            A[x] += total_prefix - prefix[start - 1]

    # output
    print(' '.join(str(A[x]) for x in range(1, N + 1)))


if __name__ == "__main__":
    main()