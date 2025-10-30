def main():
    import sys

    tokens = sys.stdin.read().strip().split()
    if not tokens:
        return

    t = int(tokens[0])
    res = []

    for i in range(1, t + 1):
        n = int(tokens[i])

        # Vanya wins immediately unless the number is already divisible by 3.
        if n % 3 == 0:
            res.append("Second")   # Vova can always return the number to a multiple of 3
        else:
            res.append("First")    # Vanya makes it divisible by 3 on his very first move

    sys.stdout.write("
".join(res))


if __name__ == "__main__":
    main()