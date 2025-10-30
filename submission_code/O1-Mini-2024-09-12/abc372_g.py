import sys, math
from math import gcd

def floor_sum(a, b, c, m):
    res = 0
    while True:
        if a == 0:
            return res + (m * (b // c))
        if a >= c or b >= c:
            res += (m - 1) * m // 2 * (a // c)
            res += m * (b // c)
            a %= c
            b %= c
        y = (a * m + b) // c
        if y == 0:
            return res
        m, b, a, c = y, a * m + b - c * y, c, a

def main():
    import sys
    import sys
    data = sys.stdin.read().split()
    idx =0
    T = int(data[idx]); idx+=1
    for _ in range(T):
        N = int(data[idx]); idx+=1
        constraints = []
        x_max = float('inf')
        for _ in range(N):
            A = int(data[idx]); B = int(data[idx+1]); C = int(data[idx+2]); idx+=3
            constraints.append( (A, B, C) )
            if A ==0:
                if C <= B:
                    x_max = -1
            else:
                temp = (C -1 - B) // A
                if temp < x_max:
                    x_max = temp
        if x_max <1:
            print(0)
            continue
        # Sort constraints by A_i / B_i ascending
        # To avoid floating, sort using log
        constraints.sort(key=lambda x: math.log(x[0]) - math.log(x[1]) if x[1]!=0 else float('inf'))
        # Build lower envelope
        env = []
        for constraint in constraints:
            A2, B2, C2 = constraint
            while len(env) >=1:
                A1, B1, C1, x1 = env[-1]
                # Compute intersection with current constraint
                delta_num = (C2 -1)*B1 - (C1 -1)*B2
                delta_den = A1 * B2 - A2 * B1
                if delta_den ==0:
                    if delta_num <=0:
                        # Current constraint is always better
                        env.pop()
                        continue
                    else:
                        break
                if delta_den <0:
                    delta_num = -delta_num
                    delta_den = -delta_den
                if delta_num <0:
                    x_cross = 0
                else:
                    if delta_num % delta_den ==0:
                        x_cross = delta_num // delta_den
                    else:
                        x_cross = delta_num // delta_den +1
                if x_cross <= x1:
                    env.pop()
                else:
                    break
            if len(env)==0:
                x_start =1
            else:
                A1, B1, C1, x1 = env[-1]
                A2, B2, C2 = constraint
                delta_num = (C2 -1)*B1 - (C1 -1)*B2
                delta_den = A1 * B2 - A2 * B1
                if delta_den ==0:
                    if (C2 -1)*B1 - (C1 -1)*B2 <0:
                        x_start =1
                    else:
                        continue
                else:
                    if delta_den <0:
                        delta_num = -delta_num
                        delta_den = -delta_den
                    if delta_num <0:
                        x_cross =0
                    else:
                        if delta_num % delta_den ==0:
                            x_cross = delta_num // delta_den
                        else:
                            x_cross = delta_num // delta_den +1
                    x_start = max(1, x_cross)
            if x_start > x_max:
                continue
            env.append( (constraint[0], constraint[1], constraint[2], x_start) )
        # Now sort env by x_start
        env.sort(key=lambda x: x[3])
        # Assign x_end for each segment
        segments = []
        for i in range(len(env)):
            A, B, C, x_start = env[i]
            if i+1 < len(env):
                x_end = env[i+1][3]-1
            else:
                x_end = x_max
            if x_start > x_end:
                continue
            segments.append( (A, B, C, x_start, x_end) )
        # Calculate the total sum
        total =0
        for seg in segments:
            A, B, C, l, r = seg
            if l > x_max:
                continue
            r = min(r, x_max)
            if l <1:
                l =1
            floor_b_c = (C -1) // B
            sum1 = (r - l +1)* floor_b_c
            sum_floor_segment = floor_sum(A, 0, B, r) - floor_sum(A, 0, B, l-1)
            total += sum1 - sum_floor_segment
        print(total)

if __name__ == "__main__":
    main()