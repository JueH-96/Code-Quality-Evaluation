def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = int(input[idx])
    idx += 1

    D = list(map(int, input[idx:idx+N]))
    idx += N

    m = N + M

    curr_low = 0
    curr_high = m - 1

    for d in D:
        a = (1 - d) % m
        allowed_low = a
        allowed_high = (a + A - 1) % m

        if allowed_low <= allowed_high:
            allowed_low_block = allowed_low
            allowed_high_block = allowed_high
            new_low = max(curr_low, allowed_low_block)
            new_high = min(curr_high, allowed_high_block)
            if new_low <= new_high:
                curr_low = new_low
                curr_high = new_high
            else:
                curr_low = None
                curr_high = None
        else:
            part1_low = max(curr_low, allowed_low_block)
            part1_high = min(curr_high, m - 1)
            part1 = (part1_low, part1_high) if part1_low <= part1_high else None

            part2_low = max(curr_low, 0)
            part2_high = min(curr_high, allowed_high_block)
            part2 = (part2_low, part2_high) if part2_low <= part2_high else None

            new_intervals = []
            if part1 is not None:
                new_intervals.append(part1)
            if part2 is not None:
                new_intervals.append(part2)

            merged = []
            if new_intervals:
                new_intervals.sort()
                current = new_intervals[0]
                for interval in new_intervals[1:]:
                    if interval[0] <= current[1] + 1:
                        current = (current[0], max(current[1], interval[1]))
                    else:
                        merged.append(current)
                        current = interval
                merged.append(current)
            else:
                merged = []

            if merged:
                curr_low, curr_high = merged[0]
            else:
                curr_low = None
                curr_high = None

    if curr_low is not None and curr_high is not None:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()