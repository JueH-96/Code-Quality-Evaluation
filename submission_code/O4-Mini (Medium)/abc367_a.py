def main():
    A, B, C = map(int, input().split())

    # If B < C, then he sleeps in the interval [B, C)
    # If B > C, then he sleeps from B to 24 and from 0 to C
    asleep = False
    if B < C:
        # straightforward interval
        if B <= A < C:
            asleep = True
    else:
        # wraps around midnight
        if A >= B or A < C:
            asleep = True

    print("No" if asleep else "Yes")


if __name__ == "__main__":
    main()