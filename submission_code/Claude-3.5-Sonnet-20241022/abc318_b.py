N = int(input())
events = []

for _ in range(N):
    a, b, c, d = map(int, input().split())
    # Add start and end x-coordinates with y-coordinates
    events.append((a, c, d, 1))  # Start of rectangle
    events.append((b, c, d, -1)) # End of rectangle

# Sort events by x-coordinate
events.sort()

total_area = 0
prev_x = events[0][0]

# For storing y-ranges
y_ranges = []

def get_covered_length():
    if not y_ranges:
        return 0
        
    # Merge overlapping y-ranges
    y_ranges.sort()
    merged = []
    start, end = y_ranges[0]
    
    for y_start, y_end in y_ranges[1:]:
        if y_start <= end:
            end = max(end, y_end)
        else:
            merged.append((start, end))
            start, end = y_start, y_end
    merged.append((start, end))
    
    # Calculate total length
    length = 0
    for start, end in merged:
        length += end - start
        
    return length

# Sweep line algorithm
for x, y1, y2, event_type in events:
    # Calculate area up to current x
    width = x - prev_x
    total_area += width * get_covered_length()
    
    # Update y-ranges
    if event_type == 1:  # Start of rectangle
        y_ranges.append((y1, y2))
    else:  # End of rectangle
        y_ranges.remove((y1, y2))
        
    prev_x = x

print(total_area)