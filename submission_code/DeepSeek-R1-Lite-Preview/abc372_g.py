import sys
from fractions import Fraction

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        lines = []
        for _ in range(N):
            A = int(data[idx])
            B = int(data[idx+1])
            C = int(data[idx+2])
            idx += 3
            m = -B / A
            b = C / A
            lines.append((m, b))
        # Sort lines by decreasing m
        lines.sort(reverse=True, key=lambda x: x[0])
        # Build convex hull
        convex_hull = []
        for line in lines:
            m, b = line
            while len(convex_hull) >= 2:
                m1, b1 = convex_hull[-2]
                m2, b2 = convex_hull[-1]
                y = Fraction(b2 - b1, m1 - m2)
                if y <= Fraction(b - b2, m2 - m):
                    convex_hull.pop()
                else:
                    break
            convex_hull.append((m, b))
        # Collect critical y values
        critical_y = []
        for i in range(1, len(convex_hull)):
            m1, b1 = convex_hull[i-1]
            m2, b2 = convex_hull[i]
            if m1 != m2:
                y = Fraction(b2 - b1, m1 - m2)
                if y > 0:
                    critical_y.append(y)
        critical_y.sort()
        # Add boundaries
        critical_y = [Fraction(0, 1)] + critical_y
        critical_y.append(Fraction('inf'))
        # Iterate through segments
        S = 0
        for i in range(len(convex_hull)):
            m, b = convex_hull[i]
            y_low = critical_y[i]
            y_high = critical_y[i+1]
            if y_low >= y_high:
                continue
            # Compute floor(m * y + b) over [y_low, y_high)
            # Since m is negative, floor is decreasing
            y_start = int(y_low) + 1
            y_end = int(y_high)
            if y_start > y_end:
                continue
            # Compute h(y) = m * y + b
            # floor(h(y_start - 1)) = floor(m * (y_start - 1) + b)
            # floor(h(y_end - 1)) = floor(m * (y_end - 1) + b)
            # Sum over y from y_start to y_end - 1 of floor(m * y + b)
            # Since m is negative, floor(h(y)) decreases by at most 1 per y
            y0 = y_start - 1
            if y0 < 0:
                y0 = 0
            h0 = m * y0 + b
            k0 = int(h0)
            dy = y_end - y_start
            S += dy * k0
        print(S)

if __name__ == '__main__':
    main()