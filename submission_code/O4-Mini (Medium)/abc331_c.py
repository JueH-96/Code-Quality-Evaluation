import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Sort the array and build prefix sums
    sA = sorted(A)
    # prefix_sums[i] = sum of sA[0..i-1]
    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i+1] = prefix_sums[i] + sA[i]
    total = prefix_sums[n]
    
    from bisect import bisect_right
    out = []
    # For each A[i], find how many elements are > A[i]
    # idx = first index in sA with value > A[i]
    # the sum of those is total - prefix_sums[idx]
    for a in A:
        idx = bisect_right(sA, a)
        out.append(str(total - prefix_sums[idx]))
    
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()