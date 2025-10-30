def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # If there is at least one 1, every zero‐block lives on a path (not a full cycle),
    # and one can choose non‐adjacent "windows" (ARC or CRA) to cover all zeros
    # without conflict.  If A is all zeros then the zero‐block wraps the whole cycle,
    # and one needs a perfect matching on the cycle, which exists iff n is even.
    if sum(A) == 0 and (n % 2) == 1:
        print("No")
    else:
        print("Yes")


if __name__ == "__main__":
    main()