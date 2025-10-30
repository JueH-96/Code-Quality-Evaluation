import sys
from bisect import bisect_left

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); Q = int(next(it))
    a = [0] * N
    for i in range(N):
        a[i] = int(next(it))
    a.sort()
    
    out = []
    INF = 10**30
    for _ in range(Q):
        b = int(next(it)); k = int(next(it))
        pos = bisect_left(a, b)
        Llen = pos
        Rlen = N - pos
        
        # If we only need the 1st closest, just compare the immediate neighbors
        if k == 1:
            left_d  = b - a[pos-1] if Llen > 0 else INF
            right_d = a[pos] - b   if Rlen > 0 else INF
            out.append(str(left_d if left_d < right_d else right_d))
            continue
        
        # Otherwise we do the "k-th element of two sorted arrays" trick
        a_start, b_start = 0, 0
        al, bl = Llen, Rlen
        K = k - 1  # zero-based target index
        
        while True:
            # If one side is exhausted, take directly from the other
            if al == 0:
                ans = a[pos + b_start + K] - b
                break
            if bl == 0:
                ans = b - a[pos - 1 - (a_start + K)]
                break
            # If we're looking for the very first element
            if K == 0:
                da = b - a[pos - 1 - a_start]
                db = a[pos + b_start] - b
                ans = da if da < db else db
                break
            
            # Choose how many elements to drop from each side
            lo = K + 1 - bl
            if lo < 1: lo = 1
            hi = al if al < K else K
            mid = (K + 1) >> 1
            if mid < lo:
                i = lo
            elif mid > hi:
                i = hi
            else:
                i = mid
            j = (K + 1) - i
            
            # Compare the i-th candidate from A0 vs the j-th from B0
            idxA = pos - a_start - i
            da = b - a[idxA]
            idxB = pos + b_start + j - 1
            db = a[idxB] - b
            
            if da < db:
                a_start += i
                al -= i
                K  -= i
            else:
                b_start += j
                bl -= j
                K  -= j
        
        out.append(str(ans))
    
    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()