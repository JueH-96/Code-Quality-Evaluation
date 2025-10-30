import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    N = int(next(it))

    water = 0        # current amount of water in liters
    prev_time = 0    # time of the previous event

    for _ in range(N):
        t = int(next(it))
        v = int(next(it))

        # water lost due to leaking since the previous event
        water = max(0, water - (t - prev_time))

        # add the new water
        water += v

        prev_time = t

    print(water)

if __name__ == "__main__":
    main()