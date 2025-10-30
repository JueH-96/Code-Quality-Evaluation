def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    N = int(data[0])
    K = int(data[1])
    groups = list(map(int, data[2:2+N]))
    
    current_seats = K
    attraction_starts = 0
    i = 0

    # Process groups until the queue is empty.
    while i < N:
        # If the current group's size is more than the available seats,
        # start the attraction and reset available seats.
        if groups[i] > current_seats:
            attraction_starts += 1
            current_seats = K
        else:
            # Otherwise, guide the group and update available seats.
            current_seats -= groups[i]
            i += 1

    # When no groups remain, start the attraction one final time.
    attraction_starts += 1

    sys.stdout.write(str(attraction_starts))

if __name__ == '__main__':
    main()