def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    start_count = 0
    empty_seats = K
    index = 0

    while index < N:
        if A[index] <= empty_seats:
            # Guide this group onto the attraction
            empty_seats -= A[index]
            index += 1
        else:
            # Start the attraction and reset empty seats
            start_count += 1
            empty_seats = K

    # When no groups remain, start the attraction one last time
    start_count += 1

    print(start_count)

# Do not forget to call main
if __name__ == "__main__":
    main()