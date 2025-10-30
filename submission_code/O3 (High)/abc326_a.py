def main():
    import sys

    # Read input
    X, Y = map(int, sys.stdin.read().split())

    diff = Y - X  # positive if going up, negative if going down

    # Determine whether stairs are used
    if (diff > 0 and diff <= 2) or (diff < 0 and -diff <= 3):
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()