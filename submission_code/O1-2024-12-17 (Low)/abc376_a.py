def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, C = map(int, data[:2])
    T = list(map(int, data[2:]))

    candies_received = 0
    last_candy_time = -1  # Track the time when the last candy was received

    for time in T:
        # First press always yields candy or if enough time has passed (>= C)
        if last_candy_time == -1 or (time - last_candy_time >= C):
            candies_received += 1
            last_candy_time = time

    print(candies_received)

# Do not forget to call main()
main()