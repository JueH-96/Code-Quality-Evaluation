import sys, bisect

def kth_distance(sorted_a, b, k):
    """
    Return the k-th smallest value of |a_i-b| for the sorted list sorted_a.
    """
    low = -1
    # the farthest point from b is always one of the ends of the sorted list
    high = max(abs(b - sorted_a[0]), abs(b - sorted_a[-1]))

    # binary-search the smallest d with at least k points inside [b-d, b+d]
    while high - low > 1:
        mid = (low + high) // 2
        cnt = bisect.bisect_right(sorted_a, b + mid) - bisect.bisect_left(sorted_a, b - mid)
        if cnt >= k:
            high = mid
        else:
            low = mid
    return high


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    N, Q = data[0], data[1]
    A = data[2:2 + N]
    A.sort()

    res, idx = [], 2 + N
    for _ in range(Q):
        b, k = data[idx], data[idx + 1]
        idx += 2
        res.append(str(kth_distance(A, b, k)))

    sys.stdout.write("
".join(res))


if __name__ == "__main__":
    main()