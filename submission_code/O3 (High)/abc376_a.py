import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    C = int(data[1])
    times = list(map(int, data[2:2 + N]))

    # At the first press a candy is always received
    candies = 1
    last_candy_time = times[0]

    for t in times[1:]:
        if t - last_candy_time >= C:
            candies += 1
            last_candy_time = t

    print(candies)

if __name__ == "__main__":
    main()