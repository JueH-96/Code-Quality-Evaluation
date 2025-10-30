import sys
import bisect
from collections import defaultdict

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def is_in_intervals(value, intervals):
    i = bisect.bisect_right(intervals, (value, float('inf'))) - 1
    if i >= 0:
        start, end = intervals[i]
        if start <= value <= end:
            return True
    return False

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    S_x = int(input[ptr]); ptr += 1
    S_y = int(input[ptr]); ptr += 1
    houses = set()
    y_to_x = defaultdict(list)
    x_to_y = defaultdict(list)
    for _ in range(N):
        X = int(input[ptr]); ptr += 1
        Y = int(input[ptr]); ptr += 1
        houses.add((X, Y))
        y_to_x[Y].append(X)
        x_to_y[X].append(Y)
    for y in y_to_x:
        y_to_x[y].sort()
    for x in x_to_y:
        x_to_y[x].sort()
    horizontal_moves = defaultdict(list)
    vertical_moves = defaultdict(list)
    x, y = S_x, S_y
    for _ in range(M):
        D = input[ptr]; ptr += 1
        C = int(input[ptr]); ptr += 1
        if D == 'U':
            new_y = y + C
            horizontal_moves[y].append((x, x))
            vertical_moves[x].append((min(y, new_y), max(y, new_y)))
            y = new_y
        elif D == 'D':
            new_y = y - C
            horizontal_moves[y].append((x, x))
            vertical_moves[x].append((min(y, new_y), max(y, new_y)))
            y = new_y
        elif D == 'L':
            new_x = x - C
            horizontal_moves[y].append((min(x, new_x), max(x, new_x)))
            vertical_moves[x].append((y, y))
            x = new_x
        elif D == 'R':
            new_x = x + C
            horizontal_moves[y].append((min(x, new_x), max(x, new_x)))
            vertical_moves[x].append((y, y))
            x = new_x
    for y in horizontal_moves:
        horizontal_moves[y] = merge_intervals(horizontal_moves[y])
    for x in vertical_moves:
        vertical_moves[x] = merge_intervals(vertical_moves[x])
    passed_houses = set()
    for (X, Y) in houses:
        if Y in horizontal_moves:
            intervals = horizontal_moves[Y]
            if is_in_intervals(X, intervals):
                passed_houses.add((X, Y))
        if X in vertical_moves:
            intervals = vertical_moves[X]
            if is_in_intervals(Y, intervals):
                passed_houses.add((X, Y))
    print(f"{x} {y} {len(passed_houses)}")

if __name__ == '__main__':
    main()