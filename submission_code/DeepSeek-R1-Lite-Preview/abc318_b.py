import sys

def merge_intervals(intervals):
    if not intervals:
        return []
    # Sort intervals by start point
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # Overlapping or adjacent intervals
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def main():
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    rectangles = []
    y_coords = set()
    for i in range(1, N + 1):
        A, B, C, D = map(int, data[i].split())
        rectangles.append((A, B, C, D))
        y_coords.add(C)
        y_coords.add(D)
    sorted_y = sorted(y_coords)
    total_area = 0
    for i in range(len(sorted_y) - 1):
        y1 = sorted_y[i]
        y2 = sorted_y[i + 1]
        # Find rectangles that cover this y-strip: C_i <= y1 < D_i
        relevant_rects = [rect for rect in rectangles if rect[2] <= y1 < rect[3]]
        x_intervals = [(rect[0], rect[1]) for rect in relevant_rects]
        merged_x = merge_intervals(x_intervals)
        width = sum(b - a for a, b in merged_x)
        height = y2 - y1
        total_area += width * height
    print(total_area)

if __name__ == "__main__":
    main()