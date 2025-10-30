import sys

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read input from stdin for better performance.
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle cases with no or invalid input, though constraints guarantee valid input.
        return

    # l[i]: max size `k` of a left-pyramid (1, ..., k) ending at index i.
    # A pyramid of size 1 can always be formed from any single element.
    l = [0] * N
    l[0] = 1
    for i in range(1, N):
        l[i] = min(l[i - 1] + 1, A[i])

    # r[i]: max size `k` of a right-pyramid (k, ..., 1) starting at index i.
    r = [0] * N
    r[N - 1] = 1
    for i in range(N - 2, -1, -1):
        r[i] = min(r[i + 1] + 1, A[i])

    # A pyramid of size `k` centered at `i` requires both a left-half of size `k`
    # and a right-half of size `k`. The max size at `i` is thus min(l[i], r[i]).
    # The final answer is the maximum possible size over all potential peak indices.
    max_k = 0
    for i in range(N):
        max_k = max(max_k, min(l[i], r[i]))

    print(max_k)

if __name__ == "__main__":
    main()