def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Mark which days have fireworks
    has_fireworks = [False] * (N + 1)
    for day in A:
        has_fireworks[day] = True

    # This array will store the next day (on or after day i) when fireworks occur
    next_firework_day = [0] * (N + 1)
    next_firework_day[N] = N  # It's guaranteed the last day (day N) has fireworks

    # Fill next_firework_day in descending order
    for i in range(N - 1, 0, -1):
        if has_fireworks[i]:
            next_firework_day[i] = i
        else:
            next_firework_day[i] = next_firework_day[i + 1]

    # Print the difference for each day
    for i in range(1, N + 1):
        print(next_firework_day[i] - i)

# Call solve()
solve()