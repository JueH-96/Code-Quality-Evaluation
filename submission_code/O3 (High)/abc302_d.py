import sys
import bisect

def max_sum_within_diff(smaller, bigger, D):
    """
    smaller : list that will be scanned linearly
    bigger  : list that is kept sorted and binary–searched
    both lists contain integers
    returns maximal a+b such that |a-b|<=D and a from smaller, b from bigger.
    If no such pair exists, returns -1.
    """
    bigger.sort()
    best = -1

    for val in smaller:
        # the largest candidate in bigger that does not exceed val+D
        idx = bisect.bisect_right(bigger, val + D) - 1
        if idx >= 0:
            b = bigger[idx]
            if abs(val - b) <= D:
                s = val + b
                if s > best:
                    best = s
    return best


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, M, D = data[:3]
    A = data[3:3 + N]
    B = data[3 + N:3 + N + M]

    # Iterate over the shorter list to minimise total binary-searches
    if N <= M:
        answer = max_sum_within_diff(A, B, D)
    else:
        answer = max_sum_within_diff(B, A, D)

    print(answer)


if __name__ == "__main__":
    main()