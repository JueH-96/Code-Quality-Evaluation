# YOUR CODE HERE
n = int(input())
events = []
for i in range(n):
    l, r = map(int, input().split())
    events.append((l, 0))  # start event
    events.append((r, 1))  # end event

# Sort events by position, with start events (0) before end events (1) at the same position
events.sort()

count = 0
active = 0
for pos, event_type in events:
    if event_type == 0:  # start event
        count += active
        active += 1
    else:  # end event
        active -= 1

print(count)