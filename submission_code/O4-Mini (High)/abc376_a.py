def main():
    import sys
    data = sys.stdin.read().split()
    N, C = map(int, data[:2])
    T = list(map(int, data[2:]))

    # Always get a candy on the first press
    count = 1
    last_time = T[0]

    # For each subsequent press, check the elapsed time
    for t in T[1:]:
        if t - last_time >= C:
            count += 1
            last_time = t

    print(count)

# Call the main function
main()