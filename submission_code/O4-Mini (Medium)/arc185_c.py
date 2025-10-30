import sys
import threading

def main():
    import sys

    data = sys.stdin.buffer.read().split()
    if not data:
        print(-1)
        return
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        print(-1)
        return
    X = int(next(it))
    A = [0]*n
    for i in range(n):
        try:
            A[i] = (int(next(it)), i)
        except StopIteration:
            A[i] = (0, i)
    # Sort by value
    A.sort(key=lambda x: x[0])
    # Two pointers for each i
    for i in range(n-2):
        ai, idx_i = A[i]
        target = X - ai
        lo = i + 1
        hi = n - 1
        while lo < hi:
            s = A[lo][0] + A[hi][0]
            if s == target:
                idx_j = A[lo][1]
                idx_k = A[hi][1]
                # sort indices to satisfy i<j<k in output
                res = sorted((idx_i+1, idx_j+1, idx_k+1))
                print(res[0], res[1], res[2])
                return
            elif s < target:
                lo += 1
            else:
                hi -= 1
    # No triple found
    print(-1)

if __name__ == "__main__":
    main()