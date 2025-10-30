import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()
index = 0
# Read N from the input
N = int(data[index])
index += 1
# Create a list to hold all events
events = []
# For each interval, add start and end events
for _ in range(N):
    l = int(data[index])
    r = int(data[index + 1])
    index += 2
    events.append((l, 0))  # Start event, type 0
    events.append((r, 1))  # End event, type 1
# Sort the events by position, then by event type (start before end)
events.sort()
# Initialize active interval count and answer
active_count = 0
answer = 0
# Process each event in sorted order
for position, event_type in events:
    if event_type == 0:  # Start event
        # Add the number of currently active intervals to the answer
        answer += active_count
        # Increment the active count
        active_count += 1
    else:  # End event
        # Decrement the active count
        active_count -= 1
# Output the answer
print(answer)