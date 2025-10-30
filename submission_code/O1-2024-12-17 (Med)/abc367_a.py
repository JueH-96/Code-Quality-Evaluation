def main():
    A, B, C = map(int, input().split())

    # If B < C, Takahashi is asleep from hour B to hour C-1 (inclusive)
    if B < C:
        asleep = (B <= A < C)
    else:
        # If B > C, Takahashi is asleep from hour B to 23, and from 0 to C-1
        asleep = (A >= B or A < C)

    print("No" if asleep else "Yes")

if __name__ == "__main__":
    main()