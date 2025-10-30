def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    seat_count = K
    count_ride = 0

    # Process each group in the queue
    for group in A:
        if seat_count < group:
            # If current seats are not enough, start the ride
            count_ride += 1
            seat_count = K  # Reset seats
        seat_count -= group

    # After all groups are seated, start the ride one last time
    count_ride += 1

    print(count_ride)

def main():
    solve()

if __name__ == "__main__":
    main()