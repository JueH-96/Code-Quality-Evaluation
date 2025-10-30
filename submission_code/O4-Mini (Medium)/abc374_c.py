import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    K = list(map(int, data[1:]))

    total = sum(K)
    half = total // 2

    # Split K into two halves
    n1 = N // 2
    left = K[:n1]
    right = K[n1:]

    # Generate all subset sums for left
    sums1 = []
    L = len(left)
    for mask in range(1 << L):
        s = 0
        # sum bits in mask
        m = mask
        idx = 0
        while m:
            # isolate lowest set bit
            lb = m & -m
            b = (lb.bit_length() - 1)
            s += left[b]
            m ^= lb
        sums1.append(s)

    # Generate all subset sums for right
    sums2 = []
    R = len(right)
    for mask in range(1 << R):
        s = 0
        m = mask
        while m:
            lb = m & -m
            b = (lb.bit_length() - 1)
            s += right[b]
            m ^= lb
        sums2.append(s)

    # Sort sums2 for binary search
    sums2.sort()

    import bisect
    best = 0
    # For each sum in sums1, find the best match in sums2 so that sum1+sum2 <= half
    for s1 in sums1:
        remain = half - s1
        if remain < 0:
            continue
        # find rightmost index in sums2 where value <= remain
        idx = bisect.bisect_right(sums2, remain) - 1
        if idx >= 0:
            candidate = s1 + sums2[idx]
            if candidate > best:
                best = candidate

    # The two group sums are best and total-best; the maximum is total-best (since best <= half)
    answer = total - best
    print(answer)


if __name__ == "__main__":
    main()