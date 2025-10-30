import sys

def main() -> None:
    """
    For each festival day i (1-indexed) output the number of days until
    the next firework day (including the current day itself if fireworks
    are launched on that day).
    """
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = list(map(int, input().split()))          # firework days (strictly increasing)

    res = []                                     # collect answers as strings
    prev = 0                                     # last firework day already processed

    for curr in A:                               # iterate over every firework day
        # For all days in the interval (prev, curr] the nearest firework is on 'curr'.
        # The distance decreases by 1 when day increases.
        for day in range(prev + 1, curr + 1):    # both ends inclusive (converted to 1-based)
            res.append(str(curr - day))
        prev = curr

    sys.stdout.write("
".join(res))

if __name__ == "__main__":
    main()