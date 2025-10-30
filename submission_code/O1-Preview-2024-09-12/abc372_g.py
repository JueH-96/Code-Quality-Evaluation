# YOUR CODE HERE
import sys
import threading
from fractions import Fraction

import bisect

def main():
    import sys
    import math
    import bisect
    import heapq
    import collections

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    T = int(sys.stdin.readline())
    for _ in range(T):
        N = int(sys.stdin.readline())
        A = []
        B = []
        C = []
        lines = []
        x_max = None
        for _ in range(N):
            a_str, b_str, c_str = sys.stdin.readline().split()
            a = int(a_str)
            b = int(b_str)
            c = int(c_str)

            # Compute x_max_i = floor((C_i - B_i -1)//A_i)
            if c - b -1 < 0:
                x_max_i = -1
            else:
                x_max_i = (c - b -1)//a
            if x_max_i < 1:
                x_max = 0
            else:
                if x_max is None or x_max_i < x_max:
                    x_max = x_max_i
            # Similarly for y_max, but we can proceed to our algorithm

            # m_i = -A_i / B_i
            m_i = Fraction(-a, b)
            c_i = Fraction(c, b)
            lines.append((m_i, c_i))
        if x_max is None or x_max <= 0:
            print(0)
            continue

        # Build lower envelope
        # Sort lines by slope descending (since m_i <=0)
        lines.sort()
        lower_envelope = []
        for m_i, c_i in lines:
            while len(lower_envelope) >= 1:
                m_prev, c_prev, x_prev = lower_envelope[-1]
                # Compute intersection point x between (m_prev, c_prev) and (m_i, c_i)
                # Intersection at x = (c_i - c_prev)/(m_prev - m_i)
                denom = m_prev - m_i
                if denom == 0:
                    # Parallel lines, keep the lower one at x approaching infinity
                    if c_i < c_prev:
                        lower_envelope.pop()
                    else:
                        break
                    continue
                x_intersect = (c_i - c_prev) / (m_prev - m_i)
                if x_intersect <= x_prev:
                    # The new line is better
                    lower_envelope.pop()
                else:
                    lower_envelope.append((m_i, c_i, x_intersect))
                    break
            else:
                # This is the first line, extends to x = +infinity
                x_intersect = Fraction(-1e20)  # Handle as negative infinity
                lower_envelope.append((m_i, c_i, x_intersect))

        # Now, for each segment in lower_envelope, process accordingly
        total = 0
        num_segments = len(lower_envelope)
        x_segments = []
        for i in range(num_segments):
            m_i, c_i, x_start = lower_envelope[i]
            if i +1 < num_segments:
                x_end = lower_envelope[i+1][2]
            else:
                x_end = Fraction(1e20)  # positive infinity
            # x ranges are x_start < x <= x_end
            # Convert x_start and x_end to integers
            # Since x >=1 and x <= x_max, we need to find the integer ranges
            # x_start may be negative infinity
            x_start = max(Fraction(1), x_start + Fraction(1, 1000000000))
            x_end = min(Fraction(x_max +1), x_end)

            if x_start > x_end:
                continue
            x_left = int(math.ceil(float(x_start)))
            x_right = int(math.floor(float(x_end))) -1
            if x_left > x_right:
                continue
            # Now, process the segment from x_left to x_right
            # y_max(x) = m_i x + c_i
            # Since m_i <0, y_max(x) decreases with x
            # Need to find x_right' where y_max(x_right') >= y_min (1)
            # Solve m_i x + c_i >= y_min
            # Since m_i <0
            b = c_i
            a = m_i
            c = Fraction(1,1)
            y_min = Fraction(1,1)
            if a * x_left + b < y_min:
                continue  # y_max(x_left) < y_min, no valid x in this segment
            x_right_ymin = int(math.floor(float((b - c*y_min)/(-a))))
            x_right_final = min(x_right, x_right_ymin)
            if x_left > x_right_final:
                continue
            n = x_right_final - x_left +1
            a_num = a.numerator
            a_den = a.denominator
            b_num = b.numerator
            b_den = b.denominator
            c_num = c.numerator
            c_den = c.denominator

            # Compute floor_sum(n, a', b', c'), where a' = a_num * c_den, b' = b_num * c_den, c' = a_den * c_num
            a_prime = a_num * c_den
            b_prime = b_num * c_den + a_num * c_den * x_left
            c_prime = a_den * c_num
            res = floor_sum(n, a_prime, b_prime, c_prime)
            total += res - (y_min -1) * n
        print(total)


def floor_sum(n, a, b, c):
    res = 0
    while True:
        if a == 0:
            return res
        if a >= c or b >= c:
            res += n * (n -1) // 2 * (a // c)
            res += n * (b // c)
            a %= c
            b %= c
        # y_max = floor((a * (n-1) + b) / c)
        y_max = (a * (n -1) + b) // c
        if y_max == 0:
            return res
        # x_max = (c * y_max - b -1) // a +1
        n_new = y_max
        a_new = c
        b_new = (c - b -1) % a
        c_new = a
        res += y_max * n
        n = n_new
        a = a_new
        b = b_new
        c = c_new
        # Continue the loop

threading.Thread(target=main).start()