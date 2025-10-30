# YOUR CODE HERE
def count_intersecting_intervals(intervals):
    events = []
    for l, r in intervals:
        events.append((l, 0))  # 0 for start
        events.append((r, 1))  # 1 for end
    events.sort()

    active = 0
    intersections = 0
    for _, event_type in events:
        if event_type == 0:  # start of interval
            intersections += active
            active += 1
        else:  # end of interval
            active -= 1

    return intersections

N = int(input())
intervals = [tuple(map(int, input().split())) for _ in range(N)]

print(count_intersecting_intervals(intervals))