import sys
import threading

def main():
    import sys
    import math

    D = int(sys.stdin.readline().strip())
    best = abs(D)  # initial: x=0,y=0

    x = 0
    while True:
        x2 = x * x
        # If x2 > D and diff keeps growing, we can stop
        if x2 > D and x2 - D > best:
            break

        rem = D - x2
        if rem >= 0:
            # y^2 should be as close to rem as possible
            y0 = math.isqrt(rem)
            # try y0 and y0+1
            for y in (y0, y0 + 1):
                diff = abs(x2 + y*y - D)
                if diff < best:
                    best = diff
        else:
            # rem < 0 => y=0 is best for this x
            diff = x2 - D
            if diff < best:
                best = diff

        x += 1

    print(best)

if __name__ == "__main__":
    main()