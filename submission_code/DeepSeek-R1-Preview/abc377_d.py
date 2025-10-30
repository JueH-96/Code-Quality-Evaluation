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
        idx += 1
        R = int(input[idx])
        idx += 1
        intervals.append((L, R))
    intervals.sort(key=lambda x: x[1])
    i = 0
    current_max = 0
    forbidden = 0
    for r in range(1, M + 1):
        while i < N and intervals[i][1] <= r:
            current_max = max(current_max, intervals[i][0])
            i += 1
        forbidden += current_max
    total = M * (M + 1) // 2
    print(total - forbidden)

if __name__ == "__main__":
    main()