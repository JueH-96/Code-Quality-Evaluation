def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, C = map(int, data[:2])
    T = list(map(int, data[2:]))

    candies = 1          # Receive a candy on the first press
    last_candy_time = T[0]

    for i in range(1, N):
        if T[i] - last_candy_time >= C:
            candies += 1
            last_candy_time = T[i]

    print(candies)

# Do not remove, needed to run the solution
if __name__ == "__main__":
    main()