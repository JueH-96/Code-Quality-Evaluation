import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    intervals = []

    for _ in range(N):
        t, d = map(int, input().split())
        intervals.append((t + d, t))   # (right end, left end)

    # sort by right end in ascending order
    intervals.sort()

    last_time = -10**30          # time we printed last product
    printed = 0

    for right, left in intervals:
        earliest = max(left, last_time + 1)  # soonest we can print here
        if earliest <= right:                # interval can accommodate it
            printed += 1
            last_time = earliest

    print(printed)

if __name__ == "__main__":
    main()