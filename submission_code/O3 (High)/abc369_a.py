def main() -> None:
    import sys

    A_str, B_str = sys.stdin.read().strip().split()
    A, B = int(A_str), int(B_str)

    possible = set()

    # A is the middle term
    possible.add(2 * A - B)

    # B is the middle term
    possible.add(2 * B - A)

    # x is the middle term (only if the average is integer)
    if (A + B) % 2 == 0:
        possible.add((A + B) // 2)

    print(len(possible))


if __name__ == "__main__":
    main()