import sys
import bisect

def main():
    import sys
    import bisect

    # Read input
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    M = int(input[idx+1])
    idx += 2
    intervals = []
    for _ in range(N):
        L = int(input[idx])
        R = int(input[idx+1])
        intervals.append((L, R))
        idx += 2

    # Sort intervals by L ascending
    intervals.sort()
    L_sorted = [intervals[i][0] for i in range(N)]
    R_sorted = [intervals[i][1] for i in range(N)]

    # Compute suffix_min
    suffix_min = [0] * (N + 1)
    suffix_min[N] = M + 1
    for j in range(N-1, -1, -1):
        suffix_min[j] = min(R_sorted[j], suffix_min[j+1])

    # Calculate total_invalid
    total_invalid = 0
    for l in range(1, M+1):
        j = bisect.bisect_left(L_sorted, l)
        if j < N:
            min_R = suffix_min[j]
        else:
            min_R = M + 1
        if min_R <= M:
            invalid_r = M - min_R + 1
            total_invalid += invalid_r

    # Calculate total_pairs and valid_pairs
    total_pairs = M * (M + 1) // 2
    valid_pairs = total_pairs - total_invalid
    print(valid_pairs)

if __name__ == '__main__':
    main()