import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
rectangles = []
index = 1
for _ in range(N):
    A, B, C, D = map(int, data[index:index+4])
    index += 4
    rectangles.append((A, B, C, D))

def calculate_covered_area(rectangles):
    events = []
    for A, B, C, D in rectangles:
        events.append((A, C, D, 1))  # Start of rectangle
        events.append((B, C, D, -1))  # End of rectangle
    events.sort()

    def calculate_interval_length(intervals):
        length = 0
        current_start = None
        for start, end in intervals:
            if current_start is None:
                current_start = start
            if end > current_start:
                length += end - current_start
                current_start = end
        return length

    prev_x = events[0][0]
    current_intervals = []
    total_area = 0

    for x, C, D, typ in events:
        if current_intervals:
            interval_length = calculate_interval_length(current_intervals)
            total_area += interval_length * (x - prev_x)

        if typ == 1:
            current_intervals.append((C, D))
            current_intervals.sort()
        else:
            current_intervals.remove((C, D))

        prev_x = x

    return total_area

S = calculate_covered_area(rectangles)
print(S)