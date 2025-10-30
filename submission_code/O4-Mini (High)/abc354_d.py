import sys
import math

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    A = int(data[0]); B = int(data[1]); C = int(data[2]); D = int(data[3])
    W = C - A
    Ht = D - B
    # Phases mod 2
    a = A & 1
    b = B & 1
    # Decompose height into full 4-blocks and remainder
    Qh = Ht // 4
    rW = W % 4
    rH = Ht % 4

    EPS = 1e-8

    def compute_S(a, b, w, h):
        # Compute ∫_{u=0..w} ∫_{v=0..h} s(u,v) du dv,
        # where s(u,v) = (-1)^{floor(a+u) + floor((b+v)/2) + floor((a+b+u+v)/2)}.
        if w <= EPS or h <= EPS:
            return 0.0
        # Gather splitting lines
        lines = []
        # vertical lines u = k - a where floor(a+u) jumps
        # k runs over integers with 0 < k-a < w  =>  a+u=k
        for k in range(a+1, a + int(w)):
            u_line = k - a
            if u_line > EPS and u_line < w - EPS:
                lines.append((0, u_line))
        # horizontal lines v = 2*m - b where floor((b+v)/2) jumps
        for m in range(1, (b + int(h))//2 + 1):
            v_line = 2*m - b
            if v_line > EPS and v_line < h - EPS:
                lines.append((1, v_line))
        # diagonal lines u+v = 2*m - (a+b)
        # m in a small range
        upper = (a + b + int(w) + int(h))//2 + 3
        for m in range(0, upper):
            c = 2*m - (a + b)
            if c > EPS and c < w + h - EPS:
                lines.append((2, c))

        # Initial rectangle polygon
        poly_list = [[(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]]

        # Clipping function: keepAbove True means keep halfspace LHS >= 0
        def clip(poly, line_type, c, keepAbove):
            new_poly = []
            n = len(poly)
            if n == 0:
                return new_poly
            for i in range(n):
                u_i, v_i = poly[i]
                u_j, v_j = poly[(i+1) % n]
                # compute signed distance
                if line_type == 0:
                    L_i = u_i - c
                    L_j = u_j - c
                elif line_type == 1:
                    L_i = v_i - c
                    L_j = v_j - c
                else:
                    L_i = u_i + v_i - c
                    L_j = u_j + v_j - c
                if keepAbove:
                    inside_i = (L_i >= -EPS)
                    inside_j = (L_j >= -EPS)
                else:
                    inside_i = (L_i <= EPS)
                    inside_j = (L_j <= EPS)
                # if vertex i is inside, keep it
                if inside_i:
                    new_poly.append((u_i, v_i))
                # if edge crosses the line, add intersection
                if inside_i != inside_j:
                    denom = (L_i - L_j)
                    if abs(denom) > EPS:
                        t = L_i / (L_i - L_j)
                        u_int = u_i + (u_j - u_i) * t
                        v_int = v_i + (v_j - v_i) * t
                        new_poly.append((u_int, v_int))
            return new_poly

        # Split polygons by each line
        for (lt, c) in lines:
            new_list = []
            for poly in poly_list:
                has_pos = False
                has_neg = False
                # check if poly spans both sides
                for (u, v) in poly:
                    if lt == 0:
                        L = u - c
                    elif lt == 1:
                        L = v - c
                    else:
                        L = u + v - c
                    if L > EPS:
                        has_pos = True
                    elif L < -EPS:
                        has_neg = True
                    if has_pos and has_neg:
                        break
                if has_pos and has_neg:
                    # split into two
                    p_plus = clip(poly, lt, c, True)
                    p_minus = clip(poly, lt, c, False)
                    if len(p_plus) >= 3:
                        new_list.append(p_plus)
                    if len(p_minus) >= 3:
                        new_list.append(p_minus)
                else:
                    # no split, keep poly as is
                    new_list.append(poly)
            poly_list = new_list

        # Integrate sign * area
        S = 0.0
        for poly in poly_list:
            area2 = 0.0
            cx_acc = 0.0
            cy_acc = 0.0
            n = len(poly)
            for i in range(n):
                x0, y0 = poly[i]
                x1, y1 = poly[(i+1) % n]
                cross = x0*y1 - x1*y0
                area2 += cross
                cx_acc += (x0 + x1) * cross
                cy_acc += (y0 + y1) * cross
            A = area2 * 0.5
            if abs(A) < EPS:
                continue
            # centroid
            cx = cx_acc / (6.0 * A)
            cy = cy_acc / (6.0 * A)
            # evaluate the parity
            p0 = math.floor(a + cx + EPS)
            p1 = math.floor((b + cy) / 2.0 + EPS)
            p2 = math.floor((a + b + cx + cy) / 2.0 + EPS)
            parity = (p0 + p1 + p2) & 1
            s_val = 1 if parity == 0 else -1
            S += A * s_val
        return S

    # Compute the two pieces of H = Qh*S1 + S3
    # S1: stripe of width rW, height 4
    if rW == 0:
        S1 = 0.0
    else:
        S1 = compute_S(a, b, float(rW), 4.0)
    # S3: corner rectangle rW x rH
    if rW == 0 or rH == 0:
        S3 = 0.0
    else:
        S3 = compute_S(a, b, float(rW), float(rH))

    Hf = Qh * S1 + S3
    # Round H to nearest integer
    if Hf >= 0:
        Hi = int(Hf + 0.5 + 1e-6)
    else:
        Hi = int(Hf - 0.5 - 1e-6)

    # Final: twice black area = area + H
    ans = W * Ht + Hi
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()