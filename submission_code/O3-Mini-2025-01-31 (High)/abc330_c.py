def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    D = int(data[0])
    
    # Initialize answer: worst-case error (for example, x = 0, y = 0 gives error = D)
    ans = D
    
    # We observe that if x^2 is significantly larger than D then the error |x^2 - D| will only increase.
    # Therefore, we only need to try x up to roughly sqrt(D) (a bit extra to be safe).
    # math.isqrt returns the integer square root.
    x_limit = math.isqrt(D) + 2  # a bit extra
    for x in range(x_limit):
        x2 = x * x
        # If x^2 exceeds D, the best candidate with this x is with y = 0, having error = x^2 - D.
        # Once that error becomes larger than our current answer, we can break out early.
        if x2 > D and (x2 - D) > ans:
            break
        
        rem = D - x2  # remainder we would like to cover with y^2
        if rem >= 0:
            # The best y is around sqrt(rem). Use math.isqrt for the floor of the square root.
            y0 = math.isqrt(rem)
            # It is possible that rounding up gives a better approximation.
            for y in (y0, y0 + 1):
                candidate = x2 + y * y
                diff = abs(candidate - D)
                if diff < ans:
                    ans = diff
                    if ans == 0:
                        # Found an exact match; no need to search further.
                        sys.stdout.write("0")
                        return
        else:
            # When rem is negative, x^2 > D, so y must be 0.
            diff = x2 - D
            if diff < ans:
                ans = diff
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()