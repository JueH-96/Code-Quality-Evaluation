import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    points = []
    index = 1
    for i in range(n):
        x = int(data[index])
        y = int(data[index+1])
        index += 2
        points.append((x, y))
    
    hull = []
    h0_frac = None

    for i in range(n):
        x_i, y_i = points[i]
        best_j_frac = None
        
        if hull:
            low = 0
            high = len(hull) - 1
            while high - low > 2:
                m1 = low + (high - low) // 3
                m2 = high - (high - low) // 3
                x1, y1 = hull[m1]
                x2, y2 = hull[m2]
                prod1 = (y_i - y1) * (x_i - x2)
                prod2 = (y_i - y2) * (x_i - x1)
                if prod1 > prod2:
                    low = m1
                else:
                    high = m2
            
            for j in range(low, high + 1):
                x_j, y_j = hull[j]
                numerator = x_i * y_j - x_j * y_i
                denominator = x_i - x_j
                if denominator == 0:
                    continue
                if best_j_frac is None:
                    best_j_frac = (numerator, denominator)
                else:
                    n_curr, d_curr = best_j_frac
                    if numerator * d_curr > n_curr * denominator:
                        best_j_frac = (numerator, denominator)
        
        if best_j_frac is not None:
            if h0_frac is None:
                h0_frac = best_j_frac
            else:
                n_curr, d_curr = h0_frac
                n_candidate, d_candidate = best_j_frac
                if n_candidate * d_curr > n_curr * d_candidate:
                    h0_frac = (n_candidate, d_candidate)
        
        while len(hull) >= 2:
            o = hull[-2]
            a = hull[-1]
            ox, oy = o
            ax, ay = a
            cross = (ax - ox) * (y_i - oy) - (ay - oy) * (x_i - ox)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append((x_i, y_i))
    
    if h0_frac is None:
        print(-1)
    else:
        n0, d0 = h0_frac
        if n0 < 0:
            print(-1)
        else:
            integer_part = n0 // d0
            remainder = n0 % d0
            decimal_digits = []
            r = remainder
            for i in range(18):
                r *= 10
                digit = r // d0
                r = r % d0
                decimal_digits.append(str(digit))
            s_integer = str(integer_part)
            s_decimal = ''.join(decimal_digits)
            print(f"{s_integer}.{s_decimal}")

if __name__ == '__main__':
    main()