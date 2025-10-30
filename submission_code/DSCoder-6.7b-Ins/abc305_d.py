import sys

def main():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    Q = int(sys.stdin.readline().strip())
    queries = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(Q)]

    sleep_intervals = []
    for i in range(N//2):
        sleep_intervals.append((A[2*i], A[2*i+1]))

    for l, r in queries:
        total_sleep = 0
        for start, end in sleep_intervals:
            if start >= l and end <= r:
                total_sleep += end - start
            elif start < l and end > r:
                total_sleep += r - l
            elif start < l and end <= r:
                total_sleep += end - l
            elif start >= l and end > r:
                total_sleep += r - start
        print(total_sleep)

if __name__ == "__main__":
    main()