def main():
    import sys

    # Read three integers from stdin
    A = list(map(int, sys.stdin.readline().split()))
    a, b, c = A

    # Check the three possible pairs
    if a * b == c or a * c == b or b * c == a:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()