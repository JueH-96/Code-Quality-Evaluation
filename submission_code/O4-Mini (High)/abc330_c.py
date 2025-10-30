import sys
import math

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    D = int(data[0])
    # Initialize answer with the worst-case: x = y = 0
    ans = D
    # We only need to try x up to floor(sqrt(D))
    max_x = math.isqrt(D)
    for x in range(max_x + 1):
        x2 = x * x
        # Remaining to match D
        rem = D - x2
        # Best y for this x is around sqrt(rem)
        y0 = math.isqrt(rem)
        # Test y0
        s = x2 + y0 * y0
        diff = s - D
        if diff < 0:
            diff = -diff
        if diff < ans:
            ans = diff
            if ans == 0:
                break
        # Also test y0 + 1
        y1 = y0 + 1
        s1 = x2 + y1 * y1
        diff1 = s1 - D
        if diff1 < 0:
            diff1 = -diff1
        if diff1 < ans:
            ans = diff1
            if ans == 0:
                break

    print(ans)

if __name__ == "__main__":
    main()