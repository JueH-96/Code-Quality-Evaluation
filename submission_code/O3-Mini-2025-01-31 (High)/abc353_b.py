def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    groups = [int(next(it)) for _ in range(N)]
    
    attraction_starts = 0
    available = K
    i = 0
    while i < N:
        if available < groups[i]:
            # Not enough seats; start the attraction and reset the available seats.
            attraction_starts += 1
            available = K
        else:
            # Seat the whole group.
            available -= groups[i]
            i += 1
    # Once all groups have been guided, start the attraction one final time.
    attraction_starts += 1

    sys.stdout.write(str(attraction_starts))
    
main()