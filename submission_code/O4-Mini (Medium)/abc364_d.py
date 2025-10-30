import sys
import threading

def main():
    import sys
    from bisect import bisect_left

    data = sys.stdin
    line = data.readline().split()
    N = int(line[0]); Q = int(line[1])
    a = list(map(int, data.readline().split()))
    a.sort()
    out = []
    for _ in range(Q):
        parts = data.readline().split()
        b = int(parts[0]); k = int(parts[1])
        # split point of a around b
        idx = bisect_left(a, b)
        L_size = idx
        R_size = N - idx
        # pointers into the virtual L and R arrays
        L_start = 0
        R_start = 0
        kk = k
        # find k-th smallest in two sorted arrays L and R
        # L[t] = b - a[idx-1 - t], t = 0..L_size-1, ascending
        # R[s] = a[idx + s] - b,      s = 0..R_size-1, ascending
        while True:
            # if L is exhausted, answer is in R
            if L_start >= L_size:
                # R[R_start + kk - 1]
                arr_idx = idx + (R_start + kk - 1)
                res = a[arr_idx] - b
                break
            # if R is exhausted
            if R_start >= R_size:
                # L[L_start + kk - 1]
                t = L_start + kk - 1
                arr_idx = idx - 1 - t
                res = b - a[arr_idx]
                break
            # if we need the 1st
            if kk == 1:
                # compare L[L_start] vs R[R_start]
                Li = idx - 1 - L_start
                Lval = b - a[Li]
                Ri = idx + R_start
                Rval = a[Ri] - b
                res = Lval if Lval <= Rval else Rval
                break
            # choose how many to compare
            # take up to kk//2 from each
            i = kk // 2
            if i > L_size - L_start:
                i = L_size - L_start
            j = kk // 2
            if j > R_size - R_start:
                j = R_size - R_start
            # indices of the last elements in these blocks
            # L block's last element index in a:
            #   t = L_start + i - 1 => arr idx = idx -1 - t = idx - L_start - i
            Li = idx - L_start - i
            Lval = b - a[Li]
            # R block's last element index: idx + R_start + j - 1
            Ri = idx + R_start + j - 1
            Rval = a[Ri] - b
            # discard the smaller block
            if Lval <= Rval:
                # discard i elements of L
                L_start += i
                kk -= i
            else:
                # discard j elements of R
                R_start += j
                kk -= j
        out.append(str(res))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()