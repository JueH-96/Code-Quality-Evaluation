import sys
import threading

def main():
    import sys
    from bisect import bisect_right

    data = sys.stdin.read().split()
    n = int(data[0])
    K = list(map(int, data[1:]))

    total = sum(K)
    half = total // 2

    # Split K into two halves
    n1 = n // 2
    n2 = n - n1
    A = K[:n1]
    B = K[n1:]

    # Generate all subset sums for A
    sumsA = [0] * (1 << n1)
    for mask in range(1, 1 << n1):
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        prev = mask ^ lsb
        sumsA[mask] = sumsA[prev] + A[i]

    # Generate all subset sums for B
    sumsB = [0] * (1 << n2)
    for mask in range(1, 1 << n2):
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        prev = mask ^ lsb
        sumsB[mask] = sumsB[prev] + B[i]

    # Sort sumsB for binary search
    sumsB.sort()

    # For each sum in sumsA, find best complement in sumsB
    best = 0
    for s1 in sumsA:
        if s1 > half:
            continue
        rem = half - s1
        # find largest s2 <= rem
        idx = bisect_right(sumsB, rem) - 1
        if idx >= 0:
            s2 = sumsB[idx]
            s = s1 + s2
            if s > best:
                best = s

    # The answer is the larger group size = total - best_smaller_half
    ans = total - best
    print(ans)

if __name__ == "__main__":
    main()