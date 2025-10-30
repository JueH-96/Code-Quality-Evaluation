import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    intervals = []
    for _ in range(N):
        l = int(input[idx])
        r = int(input[idx+1])
        intervals.append((l, r))
        idx += 2

    intervals.sort()
    L = [x[0] for x in intervals]
    R = [x[1] for x in intervals]

    non_overlapping = 0
    for i in range(N):
        target = R[i]
        # Find first index j > i where L[j] > target
        pos = bisect.bisect_right(L, target, i+1, N)
        non_overlapping += (N - pos)
    
    total = N * (N - 1) // 2
    ans = total - non_overlapping
    print(ans)

if __name__ == "__main__":
    main()