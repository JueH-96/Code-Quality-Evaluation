import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # A[l] will store the smallest R among intervals starting at l.
    # Initialize with m+1 so that if no interval starts at l,
    # it does not restrict us below the maximum possible r = m.
    A = [m+1] * (m+2)

    for _ in range(n):
        l = int(next(it))
        r = int(next(it))
        if r < A[l]:
            A[l] = r

    ans = 0
    # cur will be the minimal R among all intervals with L >= current l
    cur = m + 1
    # We process l from m down to 1, maintaining the suffix minimum of A.
    for l in range(m, 0, -1):
        if A[l] < cur:
            cur = A[l]
        # For this l, valid r are l <= r < cur,
        # so there are max(0, cur - l) choices.
        if cur > l:
            ans += (cur - l)

    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()