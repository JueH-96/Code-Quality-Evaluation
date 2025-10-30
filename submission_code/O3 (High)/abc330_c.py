import sys
import math

def main() -> None:
    D = int(sys.stdin.readline())

    # Start with the trivial pair (0, 0)
    best = D                       # |0^2 + 0^2 - D| = D
    lim  = math.isqrt(D) + 1       # we only need to try x up to ceil(sqrt(D))

    for x in range(lim + 1):
        x2 = x * x

        # If x^2 already exceeds D by more than our current best,
        # no larger x can improve the answer (y^2 is non-negative).
        if x2 - D > best:
            break

        t = D - x2
        if t >= 0:
            y0 = math.isqrt(t)     # closest integer <= sqrt(t)

            # Check the two nearest y values: y0 and y0+1
            for y in (y0, y0 + 1):
                diff = abs(x2 + y * y - D)
                if diff < best:
                    best = diff
                    if best == 0:          # perfect match found – can stop early
                        print(0)
                        return
        else:
            # t < 0  ->  x^2 already greater than D, best y is 0
            diff = x2 - D
            if diff < best:
                best = diff
                # No need to check for best == 0 here (diff can’t be 0 if t < 0)

    print(best)

if __name__ == "__main__":
    main()