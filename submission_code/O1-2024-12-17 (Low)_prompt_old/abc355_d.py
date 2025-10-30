def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    intervals = [(int(input_data[2*i+1]), int(input_data[2*i+2])) for i in range(N)]

    # We use a "line sweep" approach:
    # - Create 2N events: (coordinate, type), where type = 0 for start, 1 for end.
    # - Sort the events by (coordinate, type) so that starts come before ends if they share the same coordinate.
    # - Sweep from left to right, keeping track of how many intervals are currently "open".
    #   Each time we encounter a start, it intersects with all currently open intervals, so add that count to the result.
    #   Then increment the open count. For an end, just decrement the open count.
    # The accumulated sum is the total number of intersecting pairs.
    
    events = []
    for l, r in intervals:
        events.append((l, 0))  # 0 -> start
        events.append((r, 1))  # 1 -> end

    # Sort events so that if x is the same, start (0) comes before end (1)
    events.sort(key=lambda x: (x[0], x[1]))

    open_intervals = 0
    answer = 0
    for _, t in events:
        if t == 0:  # start of interval
            answer += open_intervals
            open_intervals += 1
        else:       # end of interval
            open_intervals -= 1

    print(answer)

# Let's call solve() to handle the problem as specified.
def main():
    solve()

if __name__ == "__main__":
    main()