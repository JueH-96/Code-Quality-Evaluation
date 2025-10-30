def main() -> None:
    import sys

    A, B, C = map(int, sys.stdin.read().strip().split())

    # Case 1: put each number in its own group (3 groups)
    if A == B == C:
        print("Yes")
        return

    # Case 2: split into two groups whose sums are equal
    total = A + B + C
    if total % 2 == 0:
        half = total // 2
        # Check every non-empty proper subset
        if (
            A == half
            or B == half
            or C == half
            or A + B == half
            or A + C == half
            or B + C == half
        ):
            print("Yes")
            return

    print("No")


if __name__ == "__main__":
    main()