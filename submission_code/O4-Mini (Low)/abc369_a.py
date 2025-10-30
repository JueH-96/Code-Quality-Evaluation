def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)

    xs = set()
    # Case 1: x is the middle term
    if (A + B) % 2 == 0:
        xs.add((A + B) // 2)
    # Case 2: x on the left
    xs.add(2 * A - B)
    # Case 3: x on the right
    xs.add(2 * B - A)

    print(len(xs))


if __name__ == "__main__":
    main()