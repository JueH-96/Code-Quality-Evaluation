import sys
import math

def floor_sum(n, m, a, b):
    res = 0
    while True:
        if a >= m:
            res += (a // m) * n * (n - 1) // 2
            a = a % m
        if b >= m:
            res += (b // m) * n
            b = b % m
        if a == 0:
            return res
        y_max = (a * (n - 1) + b) // m
        if y_max == 0:
            return res
        x_max = y_max * m - b
        res += (n - (x_max - b + a - 1) // a) * y_max
        n, m, a, b = y_max, a, m % a, (m - (x_max - b) % a) % a
        if m == 0:
            return res

def intersection_x(l1, l2):
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    numerator = B1 * (C2 - 1) - B2 * (C1 - 1)
    denominator = B1 * A2 - B2 * A1
    if denominator == 0:
        return float('inf')  # parallel lines
    return numerator / denominator

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr +=1
        lines = []
        valid = True
        for __ in range(N):
            A = int(input[ptr])
            B = int(input[ptr+1])
            C = int(input[ptr+2])
            ptr +=3
            if A + B >= C:
                valid = False
            lines.append( (A, B, C) )
        if not valid:
            print(0)
            continue
        # Compute X_max
        X_max = float('inf')
        for A, B, C in lines:
            temp = (C - B - 1) // A
            if temp < X_max:
                X_max = temp
        X_max = int(X_max)
        # Generate lines for convex hull
        # First, sort and filter lines
        sorted_lines = []
        for line in lines:
            sorted_lines.append(line)
        # Sort by A/B in decreasing order, and filter same slopes
        # Custom sort
        def line_key(line):
            A, B, C = line
            return (-A / B, - ( (C - 1) / B )) # descending slope, then descending intercept
        sorted_lines.sort(key=line_key)
        # Now, filter same slopes
        filtered = []
        for line in sorted_lines:
            if not filtered:
                filtered.append(line)
                continue
            last = filtered[-1]
            A1, B1, C1 = line
            A2, B2, C2 = last
            # Check if slopes are same
            if A1 * B2 == A2 * B1:
                # same slope, compare intercepts
                val1 = (C1 -1) * B2
                val2 = (C2 -1) * B1
                if val1 > val2:
                    filtered[-1] = line
            else:
                filtered.append(line)
        # Build convex hull
        hull = []
        for line in filtered:
            # Add line to hull, maintaining convex hull
            while len(hull) >= 2:
                l1 = hull[-2]
                l2 = hull[-1]
                # Check if l2 is redundant
                x1 = intersection_x(l1, l2)
                x2 = intersection_x(l1, line)
                if x2 <= x1:
                    hull.pop()
                else:
                    break
            hull.append(line)
        # Collect breakpoints
        break_points = []
        for i in range(1, len(hull)):
            x = intersection_x(hull[i-1], hull[i])
            break_points.append(x)
        # Generate events
        events = [1.0, float(X_max)] + break_points
        events.sort()
        # Generate intervals
        total = 0
        for i in range(len(events) - 1):
            lo = events[i]
            hi = events[i+1]
            a = math.ceil(lo)
            b = math.floor(hi - 1e-8)
            if a > X_max or b < 1:
                continue
            if a > b:
                continue
            # find active line for this interval
            x_mid = (lo + hi) / 2
            min_val = float('inf')
            active_line = None
            for line in hull:
                A_line, B_line, C_line = line
                val = (C_line - 1 - A_line * x_mid) / B_line
                if val < min_val:
                    min_val = val
                    active_line = line
            A, B, C = active_line
            K = C - 1
            m = B
            a_line = -A
            b_line = K
            sum_high = floor_sum(b + 1, m, a_line, b_line)
            sum_low = floor_sum(a, m, a_line, b_line)
            contribution = sum_high - sum_low
            total += contribution
        print(total)

if __name__ == "__main__":
    solve()