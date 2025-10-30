import sys

def solve():
    N = int(sys.stdin.readline().strip())
    intervals = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]

    total_sum = sum(intervals[i][0] for i in range(N))
    if total_sum % 2 != 0:
        print("No")
        return

    target = total_sum // 2
    intervals.sort(key=lambda x: (x[0] + x[1]))
    result = [0] * N
    for i in range(N):
        if intervals[i][0] <= target <= intervals[i][1]:
            result[i] = target
            target -= intervals[i][0]
        else:
            result[i] = intervals[i][1]
            target -= intervals[i][1]

    if target != 0:
        print("No")
    else:
        print("Yes")
        print(" ".join(map(str, result)))

solve()