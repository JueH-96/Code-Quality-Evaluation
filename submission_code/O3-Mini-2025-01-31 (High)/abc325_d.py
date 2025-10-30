def main():
    import sys

    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Number of products
    N = int(data[0])
    intervals = [None] * N

    # Each product i has an available interval [T, T+D]
    # where T is the time it enters, and D is the duration during which it is in range.
    # So the product i is available from time T to time T+D.
    # We store intervals as (finish, start), where finish = T + D.
    index = 1
    for i in range(N):
        T = int(data[index])
        D = int(data[index + 1])
        index += 2
        intervals[i] = (T + D, T)

    # Sort intervals in order of their end time.
    intervals.sort(key=lambda x: (x[0], x[1]))

    count = 0
    # current_time is the earliest time that the printer is available to print.
    # Initially, we set it to a very small number so that for the first product,
    # the printing time will be max(current_time, T) = T.
    current_time = -10**20

    # For each product, try to schedule printing on it.
    # The printer can print at any time t provided that:
    #    t is at least the product's start time T,
    #    t does not exceed the finish time (T+D), and
    #    t is at least current_time (which is last printing time + 1).
    for finish, start in intervals:
        # The earliest time we can print on this product is:
        t = max(current_time, start)
        if t <= finish:
            count += 1
            # After printing at time t, the printer requires a charge time of 1 microsecond.
            # Therefore, the next printing event can occur at time t+1.
            current_time = t + 1

    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()