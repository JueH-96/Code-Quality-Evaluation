import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    q = int(data[1])
    A = list(map(int, data[2:2+n]))
    A.sort()
    queries = []
    idx = 2 + n
    for i in range(q):
        b = int(data[idx])
        k = int(data[idx+1])
        idx += 2
        queries.append((b, k))
    
    res = []
    for b, k in queries:
        max_dist = max(b - A[0], A[-1] - b)
        lo = 0
        hi = max_dist
        while lo < hi:
            mid = (lo + hi) // 2
            left_bound = b - mid
            right_bound = b + mid
            l_idx = bisect.bisect_left(A, left_bound)
            r_idx = bisect.bisect_right(A, right_bound)
            count = r_idx - l_idx
            if count >= k:
                hi = mid
            else:
                lo = mid + 1
        res.append(str(lo))
    
    print("
".join(res))

if __name__ == "__main__":
    main()