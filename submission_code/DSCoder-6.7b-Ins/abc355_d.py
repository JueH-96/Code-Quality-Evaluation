import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    intervals = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    return N, intervals

def solve(N, intervals):
    intervals.sort()
    count = 0
    last_end = -1
    for start, end in intervals:
        if last_end >= start:
            count += 1
            last_end = max(last_end, end)
        else:
            last_end = end
    return count

def main():
    N, intervals = read_input()
    result = solve(N, intervals)
    print(result)

if __name__ == "__main__":
    main()