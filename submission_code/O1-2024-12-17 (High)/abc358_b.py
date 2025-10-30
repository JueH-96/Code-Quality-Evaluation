def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, A = map(int, data[:2])
    T = list(map(int, data[2:]))

    current_time = 0
    for arrival in T:
        # If the booth is free when the person arrives (arrival >= current_time),
        # they start right away. Otherwise, they have to wait until current_time.
        start_time = max(arrival, current_time)
        finish_time = start_time + A
        print(finish_time)
        current_time = finish_time

# Do not forget to call main()
if __name__ == "__main__":
    main()