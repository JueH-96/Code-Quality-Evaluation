import sys

def main() -> None:
    input = sys.stdin.readline
    N, K = map(int, input().split())

    events = {}              # day -> total change of pills from that day
    total = 0                # pills on day 1

    for _ in range(N):
        a, b = map(int, input().split())
        total += b
        stop_day = a + 1     # from this day the medicine is no longer taken
        events[stop_day] = events.get(stop_day, 0) - b

    # day 1
    if total <= K:
        print(1)
        return

    # process days in which something changes
    for day in sorted(events):
        total += events[day]     # medicines that end on this day
        if total <= K:
            print(day)
            return

if __name__ == "__main__":
    main()