import sys
import threading

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = list(map(int, data[1:1+n]))
    
    # Two‐pointer greedy to count the maximum pairs.
    # i scans the smaller half, j scans the larger half.
    i = 0
    j = n // 2
    count = 0
    # We only ever need at most floor(n/2) small items.
    while i < n // 2 and j < n:
        # If A[i] can go on top of A[j], form a pair.
        if 2 * A[i] <= A[j]:
            count += 1
            i += 1
            j += 1
        else:
            # Otherwise try a bigger base mochi.
            j += 1
    
    print(count)

if __name__ == "__main__":
    main()