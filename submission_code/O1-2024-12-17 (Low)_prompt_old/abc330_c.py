import sys
import math

def solve():
    D = int(sys.stdin.readline())
    
    # We will search for x up to sqrt(D) + 1
    # For each x, we compute T = D - x^2.
    # If T >= 0, then y is around sqrt(T).
    # We check y = floor(sqrt(T)) and y = floor(sqrt(T)) + 1 (to handle rounding)
    # If T < 0, we only check y = 0.
    #
    # We track the minimum absolute difference |x^2 + y^2 - D|.
    
    M = int(math.isqrt(D)) + 1
    ans = D  # a trivial upper bound (e.g., x=0,y=0 => difference = D)
    
    for x in range(M + 1):
        xx = x*x
        # Case y=0
        diff = abs(xx - D)
        if diff < ans:
            ans = diff
            if ans == 0:
                print(0)
                return
        
        T = D - xx
        if T >= 0:
            y0 = int(math.isqrt(T))
            # Check y0 and y0+1
            for y in (y0, y0+1):
                yy = y*y
                candidate = abs(xx + yy - D)
                if candidate < ans:
                    ans = candidate
                    if ans == 0:
                        print(0)
                        return
    
    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()