import sys
import math
from fractions import Fraction

# An efficient implementation of the floor sum algorithm.
# It computes sum_{i=0}^{n-1} floor((a * i + b) / m) for integers n, m, a, b.
# It correctly handles negative values for a and b.
def floor_sum(n, m, a, b):
    if n == 0:
        return 0
    
    res = 0
    
    # Handle negative 'a' by transforming the problem.
    # floor((a*i+b)/m) where a < 0 can be rewritten using floor(-z) = -ceil(z).
    # ceil(x/y) = (x+y-1)//y for integer x, y with y>0.
    # So, floor(-z/m) = -ceil(z/m) = -((z+m-1)//m).
    # sum floor((-A*i+b)/m) = -sum floor((A*i-b+m-1)/m) for A>0.
    if a < 0:
        a_p = -a
        b_p = -b
        return -floor_sum(n, m, a_p, b_p + m - 1)
    
    # Reduce 'a' and 'b' modulo 'm'.
    # floor((a*i+b)/m) = floor(((a%m)*i + b%m)/m) + (a//m)*i + (b//m)
    if a >= m:
        res += n * (n - 1) // 2 * (a // m)
        a %= m
    
    if b >= m:
        res += n * (b // m)
        b %= m
        
    # Now 0 <= a < m and 0 <= b < m.
    # The core of the algorithm, based on reciprocity.
    y_max = (a * (n - 1) + b) // m
    if y_max == 0:
        return res
        
    x_max = y_max * m - b
    res += (n - (x_max + a - 1) // a) * y_max
    res += floor_sum(y_max, a, m, (a - x_max % a) % a)
    return res

def solve():
    """
    Solves a single test case.
    """
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        lines_info = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    except (IOError, ValueError):
        return

    # For a pair (x, y) to exist, (1, 1) must be a potential candidate.
    # This means A_i * 1 + B_i * 1 < C_i for all i.
    for A, B, C in lines_info:
        if A + B >= C:
            print(0)
            return

    # 1. Filter lines: For each slope, keep only the one with the minimum intercept.
    slopes = {}
    for A, B, C in lines_info:
        g = math.gcd(A, B)
        key = (A // g, B // g)
        
        if key not in slopes:
            slopes[key] = (A, B, C)
        else:
            A_j, B_j, C_j = slopes[key]
            # Compare intercepts (C-1)/B vs (C_j-1)/B_j for the lower envelope.
            if (C - 1) * B_j < (C_j - 1) * B:
                slopes[key] = (A, B, C)

    unique_lines = list(slopes.values())
    
    # 2. Sort by slope -A/B ascending, which means A/B descending.
    unique_lines.sort(key=lambda L: Fraction(L[0], L[1]), reverse=True)

    # 3. Build the lower envelope (convex hull trick).
    hull = []
    for L_k in unique_lines:
        while len(hull) >= 2:
            L_j = hull[-1]
            L_i = hull[-2]
            # Check for convexity. If L_j is made redundant by L_k, pop L_j.
            # This happens if the intersection of L_i,L_j is to the right of L_j,L_k
            num1 = (L_j[2] - 1) * L_i[1] - (L_i[2] - 1) * L_j[1]
            den1 = L_j[0] * L_i[1] - L_i[0] * L_j[1]
            num2 = (L_k[2] - 1) * L_j[1] - (L_j[2] - 1) * L_k[1]
            den2 = L_k[0] * L_j[1] - L_j[0] * L_k[1]

            if num1 * den2 >= num2 * den1:
                hull.pop()
            else:
                break
        hull.append(L_k)

    # 4. Sum counts over segments of the lower envelope.
    total_count = 0
    current_x = 1
    
    # Find which segment of the hull is active at x=1
    start_k = 0
    if len(hull) > 1:
        min_y = float('inf')
        for i, (A, B, C) in enumerate(hull):
            y = (C - 1 - A * current_x) / B
            if y < min_y:
                min_y = y
                start_k = i

    for k in range(start_k, len(hull)):
        A, B, C = hull[k]
        
        y_at_current_x = (C - 1 - A * current_x) // B
        if y_at_current_x < 1:
            break
            
        x_end = 0
        if k == len(hull) - 1:
            # Last segment, extends until y drops below 1.
            # A*x + B*1 < C => A*x < C-B => x <= (C-B-1)/A
            if C - B <= 0:
                x_end = 0
            else:
                x_end = (C - B - 1) // A
        else:
            # Segment ends at the intersection with the next line on the hull.
            A_next, B_next, C_next = hull[k + 1]
            num = (C - 1) * B_next - (C_next - 1) * B
            den = A * B_next - A_next * B
            x_end = num // den
            
        x_start = current_x
        if x_start > x_end:
            continue
            
        # Sum y(x) for x in [x_start, x_end] using floor_sum
        n_fs = x_end - x_start + 1
        m_fs = B
        a_fs = -A
        b_fs = C - 1 - A * x_start
        
        total_count += floor_sum(n_fs, m_fs, a_fs, b_fs)
        
        current_x = x_end + 1
        
    print(total_count)

def main():
    try:
        T_str = sys.stdin.readline()
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()