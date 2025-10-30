import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0])
    A = list(map(int, data.readline().split()))
    if N == 0:
        print(0)
        return

    maxA = max(A)
    # Build positions list: pos[v] = sorted list of indices i where A[i] == v
    pos = [[] for _ in range(maxA + 1)]
    for i, v in enumerate(A, start=1):
        pos[v].append(i)

    res = 0
    # For each value x present, count subarrays where x appears and x-1 does not
    for x in range(1, maxA + 1):
        a = pos[x]
        if not a:
            continue
        b_list = pos[x - 1]  # positions of x-1
        pa = 0
        a_len = len(a)
        b_prev = 0
        # Process segments between occurrences of x-1
        for b in b_list:
            l = b_prev + 1
            r = b - 1
            if l <= r:
                seg_len = r - l + 1
                total_seg = seg_len * (seg_len + 1) // 2
                prev = l - 1
                gaps = 0
                # Erase all gaps contributed by positions of x in [l, r]
                while pa < a_len and a[pa] <= r:
                    gap = a[pa] - prev - 1
                    gaps += gap * (gap + 1) // 2
                    prev = a[pa]
                    pa += 1
                # Final gap after last x
                gap = r - prev
                gaps += gap * (gap + 1) // 2
                res += total_seg - gaps
            b_prev = b

        # Last segment after the last x-1 up to N
        l = b_prev + 1
        r = N
        if l <= r:
            seg_len = r - l + 1
            total_seg = seg_len * (seg_len + 1) // 2
            prev = l - 1
            gaps = 0
            while pa < a_len and a[pa] <= r:
                gap = a[pa] - prev - 1
                gaps += gap * (gap + 1) // 2
                prev = a[pa]
                pa += 1
            gap = r - prev
            gaps += gap * (gap + 1) // 2
            res += total_seg - gaps

    # Output the total sum of f(L,R)
    sys.stdout.write(str(res))


if __name__ == "__main__":
    main()