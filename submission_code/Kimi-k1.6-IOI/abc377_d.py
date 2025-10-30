def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    intervals = []
    for _ in range(N):
        L = int(input[idx])
        R = int(input[idx+1])
        intervals.append((L, R))
        idx += 2
    # Sort intervals by their R value
    intervals.sort(key=lambda x: x[1])
    current_max_L = 0
    idx_interval = 0
    invalid = 0
    for r in range(1, M+1):
        # Process all intervals with R <= current r
        while idx_interval < N and intervals[idx_interval][1] <= r:
            current_max_L = max(current_max_L, intervals[idx_interval][0])
            idx_interval += 1
        invalid += min(current_max_L, r)
    total = M * (M + 1) // 2
    print(total - invalid)

if __name__ == "__main__":
    main()