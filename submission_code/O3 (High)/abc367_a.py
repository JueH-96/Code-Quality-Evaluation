import sys

def main() -> None:
    A, B, C = map(int, sys.stdin.readline().split())

    t = B                       # time when Takahashi falls asleep
    while t != C:               # loop over all sleeping hours (C is the wake-up time, exclusive)
        if t == A:              # if shouting time falls inside the sleeping interval
            print("No")
            return
        t = (t + 1) % 24        # move to next hour in a circular manner
    print("Yes")                # never asleep at A, so he can shout every day


if __name__ == "__main__":
    main()