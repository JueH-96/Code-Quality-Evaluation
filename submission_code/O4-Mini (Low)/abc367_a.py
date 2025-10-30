def main():
    A, B, C = map(int, input().split())
    # He sleeps from B to C (wrapping around midnight if B > C).
    # Check if A (the shout time) falls within the sleeping interval.
    asleep = False
    if B < C:
        # Sleep is a contiguous interval [B, C)
        asleep = (B <= A < C)
    else:
        # Sleep wraps around midnight: [B,24) and [0,C)
        asleep = (A >= B or A < C)
    print("No" if asleep else "Yes")

if __name__ == "__main__":
    main()