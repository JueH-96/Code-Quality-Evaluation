import sys

def orientation(a, b, c):
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if val == 0:
        return 0  # colinear, should not happen per problem constraints
    return 1 if val > 0 else 2  # 1 for clockwise, 2 for counter-clockwise

def segments_intersect(a, b, c, d):
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    p_list = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        p_list.append((x, y))
        idx += 2
    
    q_list = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        q_list.append((x, y))
        idx += 2
    
    # Sort P and Q points by x, then y
    sorted_p = sorted(enumerate(p_list), key=lambda x: (x[1][0], x[1][1]))
    sorted_q = sorted(enumerate(q_list), key=lambda x: (x[1][0], x[1][1]))
    
    used = [False] * N
    r = [0] * N
    segments = []
    
    valid = True
    for i in range(N):
        p_orig_idx, p = sorted_p[i]
        found = False
        for j in range(N):
            if not used[j]:
                q_orig_idx, q = sorted_q[j]
                # Check if this segment intersects with any existing segments
                intersect = False
                for seg in segments:
                    existing_p, existing_q = seg
                    if segments_intersect(p, q, existing_p, existing_q):
                        intersect = True
                        break
                if not intersect:
                    used[j] = True
                    r[p_orig_idx] = q_orig_idx + 1  # Convert to 1-based index
                    segments.append((p, q))
                    found = True
                    break
        if not found:
            valid = False
            break
    
    if valid:
        print(' '.join(map(str, r)))
    else:
        print(-1)

if __name__ == '__main__':
    main()