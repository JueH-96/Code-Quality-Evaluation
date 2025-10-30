import sys
import bisect

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    K = [int(next(it)) for _ in range(n)]
    total = sum(K)
    half = total // 2

    # Split into two halves
    n1 = n // 2
    arr1 = K[:n1]
    arr2 = K[n1:]

    # Generate all subset sums for each half
    sums1 = [0]
    for x in arr1:
        sums1 += [v + x for v in sums1]

    sums2 = [0]
    for x in arr2:
        sums2 += [v + x for v in sums2]
    sums2.sort()

    best_low = 0      # best sum <= half
    best_high = total # best sum >= half

    # For each partial sum in sums1, binary search in sums2
    for v in sums1:
        target = half - v

        # find best <= half combination
        idx = bisect.bisect_right(sums2, target) - 1
        if idx >= 0:
            s = v + sums2[idx]
            if s > best_low:
                best_low = s

        # find best >= half combination
        idx = bisect.bisect_left(sums2, target)
        if idx < len(sums2):
            s = v + sums2[idx]
            if s < best_high:
                best_high = s

    # The answer is the minimum possible max(sum_A, sum_B)
    # For sums <= half, the max is total - best_low; for sums >= half, it's best_high
    ans = min(total - best_low, best_high)
    print(ans)

if __name__ == "__main__":
    main()