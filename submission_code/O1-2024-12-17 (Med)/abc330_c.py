def main():
    import sys
    import math

    D = int(sys.stdin.readline())

    # We can use a two-pointer-like approach. Let x = 0 and y = floor(sqrt(D)) + 1.
    # Then, while x <= y, compute s = x^2 + y^2.
    #  - If s < D, increment x (to try making s larger).
    #  - If s > D, decrement y (to try making s smaller).
    # Keep track of the minimal |s - D|.
    # Because x^2 + y^2 = y^2 + x^2 is symmetric, restricting x <= y covers all possible pairs.
    
    x = 0
    y = int(math.isqrt(D)) + 1  # a bit beyond sqrt(D) to be safe
    x_sq = 0
    y_sq = y * y

    best = abs(D - y_sq)  # initial difference (with x=0)

    while x <= y:
        s = x_sq + y_sq
        diff = abs(s - D)
        if diff < best:
            best = diff
        if s == D:
            best = 0
            break
        if s < D:
            x += 1
            x_sq = x * x
        else:
            y -= 1
            y_sq = y * y

    print(best)

# don't forget to call main at the end
if __name__ == "__main__":
    main()