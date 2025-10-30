def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, C = map(int, data[:2])
    T = list(map(int, data[2:]))

    candies = 0
    last_time = -10**18  # initialize to a very small time

    for t in T:
        if t - last_time >= C:
            candies += 1
            last_time = t

    print(candies)

if __name__ == "__main__":
    main()