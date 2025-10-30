def main():
    import sys
    data = sys.stdin.read().split()
    A, B = map(int, data)

    # We want all x such that A, B, and x can be arranged into an arithmetic sequence.
    # An arithmetic sequence of three elements (p, q, r) satisfies 2*q = p + r.
    # Thus, one of A, B, x must be the middle term. We get these conditions:
    #   1) A is the middle:   x = 2*A - B
    #   2) B is the middle:   x = 2*B - A
    #   3) x is the middle:   x = (A + B) / 2  (must be integer)
    #
    # We gather these possible x values, remove duplicates, and count them.

    possible = set()
    possible.add(2*A - B)     # A is middle
    possible.add(2*B - A)     # B is middle
    if (A + B) % 2 == 0:      # x is middle (only when A+B is even)
        possible.add((A + B)//2)

    print(len(possible))

# Do not forget to call main().
if __name__ == "__main__":
    main()