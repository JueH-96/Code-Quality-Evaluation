def main() -> None:
    import sys

    # Read inputs
    N, X, Y, Z = map(int, sys.stdin.read().split())

    # Z is on the route from X to Y iff it lies strictly between them.
    if (X < Z < Y) or (Y < Z < X):
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()