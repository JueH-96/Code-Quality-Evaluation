import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    # col[i]: color at cell i (1-indexed)
    col = list(range(N+2))  # col[1]=1,...,col[N]=N. col[0], col[N+1] sentinels
    # count[c]: how many cells currently painted color c
    cnt = [0]*(N+2)
    for i in range(1, N+1):
        cnt[i] = 1

    # breakpoints: positions i where col[i] != col[i+1]
    # maintain sorted list of breakpoints between 1..N-1
    breaks = list(range(1, N))
    from bisect import bisect_left, bisect_right, insort

    ans = []
    for _ in range(Q):
        q = input().split()
        if not q:
            q = input().split()
        t = int(q[0])
        if t == 1:
            x = int(q[1])
            c = int(q[2])
            oldc = col[x]
            if oldc == c:
                continue
            # find segment [l..r] containing x
            # left boundary: last break < x
            i = bisect_left(breaks, x)
            if i == 0:
                l = 1
            else:
                l = breaks[i-1] + 1
            # right boundary: first break >= x
            j = bisect_right(breaks, x)
            if j == len(breaks):
                r = N
            else:
                r = breaks[j]
            length = r - l + 1
            # update counts
            cnt[oldc] -= length
            cnt[c] += length
            # remove internal breaks in [l, r-1]
            lo = bisect_left(breaks, l)
            hi = bisect_right(breaks, r-1)
            # delete breaks[lo:hi]
            if lo < hi:
                del breaks[lo:hi]
            # now handle borders
            # left border at l-1
            if l > 1:
                bp = l-1
                # if col[l-1] == new c => no break, else ensure break
                if col[l-1] == c:
                    # remove break if exists
                    k = bisect_left(breaks, bp)
                    if k < len(breaks) and breaks[k] == bp:
                        breaks.pop(k)
                else:
                    # insert if missing
                    k = bisect_left(breaks, bp)
                    if k >= len(breaks) or breaks[k] != bp:
                        breaks.insert(k, bp)
            # right border at r
            if r < N:
                bp = r
                if col[r+1] == c:
                    k = bisect_left(breaks, bp)
                    if k < len(breaks) and breaks[k] == bp:
                        breaks.pop(k)
                else:
                    k = bisect_left(breaks, bp)
                    if k >= len(breaks) or breaks[k] != bp:
                        breaks.insert(k, bp)
            # finally recolor the segment in col[]
            # NOTE: This is O(length). Hope it passes.
            for i2 in range(l, r+1):
                col[i2] = c

        else:
            c = int(q[1])
            ans.append(str(cnt[c]))
    sys.stdout.write("
".join(ans))

if __name__ == "__main__":
    main()