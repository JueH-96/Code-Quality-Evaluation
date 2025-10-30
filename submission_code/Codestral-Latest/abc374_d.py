import math
import itertools

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def travel_time(x1, y1, x2, y2, S, T):
    return distance(x1, y1, x2, y2) / T

def move_time(x1, y1, x2, y2, S):
    return distance(x1, y1, x2, y2) / S

def total_time(segments, S, T):
    min_time = float('inf')
    for order in itertools.permutations(segments):
        current_time = 0
        current_x, current_y = 0, 0
        for (x1, y1, x2, y2) in order:
            if (current_x, current_y) != (x1, y1):
                current_time += move_time(current_x, current_y, x1, y1, S)
            current_time += travel_time(x1, y1, x2, y2, S, T)
            current_x, current_y = x2, y2
        min_time = min(min_time, current_time)
    return min_time

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = int(data[1])
    T = int(data[2])

    segments = []
    for i in range(N):
        A, B, C, D = map(int, data[3 + 4 * i: 3 + 4 * (i + 1)])
        segments.append((A, B, C, D))

    result = total_time(segments, S, T)
    print(result)

if __name__ == "__main__":
    main()