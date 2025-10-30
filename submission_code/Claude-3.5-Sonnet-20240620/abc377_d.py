# YOUR CODE HERE
def count_valid_pairs(N, M, intervals):
    events = []
    for L, R in intervals:
        events.append((L, 0))  # 0 for start of interval
        events.append((R + 1, 1))  # 1 for end of interval
    events.sort()

    active_intervals = 0
    last_pos = 0
    total_pairs = 0
    valid_positions = 0

    for pos, event_type in events:
        if pos > M:
            break

        if active_intervals == 0:
            valid_positions += pos - last_pos

        if event_type == 0:  # Start of interval
            if active_intervals == 0:
                total_pairs += (valid_positions * (valid_positions + 1)) // 2
                valid_positions = 0
            active_intervals += 1
        else:  # End of interval
            active_intervals -= 1

        last_pos = pos

    if active_intervals == 0:
        valid_positions += M - last_pos + 1
        total_pairs += (valid_positions * (valid_positions + 1)) // 2

    return total_pairs

# Read input
N, M = map(int, input().split())
intervals = [tuple(map(int, input().split())) for _ in range(N)]

# Solve and print output
result = count_valid_pairs(N, M, intervals)
print(result)