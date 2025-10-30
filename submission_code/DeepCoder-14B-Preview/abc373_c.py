def orientation(a, b, c):
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if val == 0:
        return 0  # colinear
    return 1 if val > 0 else 2  # clock or counter clock wise

def segments_intersect(a, b, c, d):
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True
    return False

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1

    P = []
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr+1])
        P.append((x, y))
        ptr += 2

    Q = []
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr+1])
        Q.append((x, y))
        ptr += 2

    original_Q = [ (q[0], q[1], i+1) for i, q in enumerate(Q) ]

    possible_keys = [
        lambda x: (x[0], x[1]),
        lambda x: (x[1], x[0]),
        lambda x: (-x[0], x[1]),
        lambda x: (x[0], -x[1]),
        lambda x: (-x[0], -x[1]),
        lambda x: (x[1], -x[0]),
    ]

    for key in possible_keys:
        sorted_Q = sorted(original_Q, key=key)
        R = [ q[2] for q in sorted_Q ]
        
        segments = []
        for i in range(N):
            p = P[i]
            q = Q[R[i] - 1]
            segments.append( (p, q) )
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                a = segments[i][0]
                b = segments[i][1]
                c = segments[j][0]
                d = segments[j][1]
                if segments_intersect(a, b, c, d):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            print(' '.join(map(str, R)))
            return
    
    print(-1)

if __name__ == "__main__":
    main()