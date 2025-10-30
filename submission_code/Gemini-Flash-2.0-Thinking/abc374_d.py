import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve():
    n, s, t = map(int, input().split())
    segments = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        segments.append(((a, b), (c, d)))

    memo = {}

    def find_min_time(current_pos, remaining_segments):
        if not remaining_segments:
            return 0

        state = (current_pos, tuple(sorted(remaining_segments)))
        if state in memo:
            return memo[state]

        min_time = float('inf')

        for i in range(len(remaining_segments)):
            segment_index = remaining_segments[i]
            p1, p2 = segments[segment_index]

            remaining = list(remaining_segments)
            remaining.pop(i)
            remaining = tuple(sorted(remaining))

            # Start at p1
            move_time_p1 = dist(current_pos, p1) / s
            print_time = dist(p1, p2) / t
            time_p1 = move_time_p1 + print_time
            remaining_time_p1 = find_min_time(p2, tuple(remaining))
            min_time = min(min_time, time_p1 + remaining_time_p1)

            # Start at p2
            move_time_p2 = dist(current_pos, p2) / s
            print_time = dist(p2, p1) / t
            time_p2 = move_time_p2 + print_time
            remaining_time_p2 = find_min_time(p1, tuple(remaining))
            min_time = min(min_time, time_p2 + remaining_time_p2)

        memo[state] = min_time
        return min_time

    initial_segments = tuple(range(n))
    result = find_min_time((0, 0), initial_segments)
    print(result)

solve()