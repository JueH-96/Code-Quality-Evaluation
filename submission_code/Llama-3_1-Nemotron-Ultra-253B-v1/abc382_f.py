import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    N = int(input[idx]); idx +=1

    bars = []
    for _ in range(N):
        R = int(input[idx]); idx +=1
        C = int(input[idx]); idx +=1
        L = int(input[idx]); idx +=1
        bars.append( (R, C, L) )

    blocked = dict()
    res = [0] * N
    for i in reversed(range(N)):
        R_i, C_i, L_i = bars[i]
        a = C_i
        b = C_i + L_i -1

        low = R_i
        high = H
        best = R_i
        while low <= high:
            mid = (low + high) // 2
            intervals = blocked.get(mid, [])
            found = False
            if intervals:
                pos = bisect.bisect_right(intervals, (b, float('inf'))) -1
                if pos >= 0:
                    s, e = intervals[pos]
                    if e >= a:
                        found = True
                if not found and pos +1 < len(intervals):
                    s, e = intervals[pos+1]
                    if s <= b and e >= a:
                        found = True
            if not found:
                best = mid
                low = mid +1
            else:
                high = mid -1
        r = best
        res[i] = r
        if r not in blocked:
            blocked[r] = []
        bisect.insort(blocked[r], (a, b))

    for r in res:
        print(r)

if __name__ == '__main__':
    main()