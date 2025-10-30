# YOUR CODE HERE
N = int(input())
intervals = []
for i in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Prepare events for line sweep
events = []
for i, (l, r) in enumerate(intervals):
    events.append((l, 0, i))  # Type 0 for start of interval
    events.append((r, 1, i))  # Type 1 for end of interval

# Sort events by position, then by event type (start before end)
events.sort()

active_intervals = set()
count = 0

for pos, event_type, interval_idx in events:
    if event_type == 0:  # Start of interval
        # Count intersections with all currently active intervals
        count += len(active_intervals)
        active_intervals.add(interval_idx)
    else:  # End of interval
        active_intervals.remove(interval_idx)

print(count)