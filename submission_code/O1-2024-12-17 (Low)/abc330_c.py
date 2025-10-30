import sys
import math

def main():
    D = int(sys.stdin.readline().strip())
    # We'll search over x in [0..isqrt(D)+2].
    # For each x, if x^2 <= D, we consider y around sqrt(D - x^2),
    # checking y0, y0+1, (y0-1 if >=0). If x^2 > D, we only check y=0.
    # We track the minimum absolute difference |x^2 + y^2 - D|.
    
    limit = math.isqrt(D) + 2
    min_diff = D  # start with a max that can't be exceeded by 0^2 + 0^2
    
    for x in range(limit+1):
        x2 = x*x
        if x2 <= D:
            t = D - x2
            y0 = math.isqrt(t)
            candidates = [y0, y0+1]
            if y0 > 0:
                candidates.append(y0-1)
            for y in candidates:
                diff = abs(x2 + y*y - D)
                if diff < min_diff:
                    min_diff = diff
                if min_diff == 0:
                    print(0)
                    return
        else:
            # x^2 already exceeds D, best y is 0
            diff = x2 - D
            if diff < min_diff:
                min_diff = diff
            # no need to consider larger y since x^2 + y^2 would only grow
            if min_diff == 0:
                print(0)
                return
    
    print(min_diff)

if __name__ == "__main__":
    main()