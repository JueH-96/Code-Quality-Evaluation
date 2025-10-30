import math

def orientation(p1, p2, p3):
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if val == 0:
        return 0
    return 1 if val > 0 else -1

def segments_intersect(seg1, seg2):
    p1, q1 = seg1
    p2, q2 = seg2
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    return False

def solve():
    n = int(input())
    p_points = []
    for _ in range(n):
        x, y = map(int, input().split())
        p_points.append((x, y))
    q_points = []
    for _ in range(n):
        x, y = map(int, input().split())
        q_points.append((x, y))
    
    min_x = min(p[0] for p in p_points) + min(q[0] for q in q_points)
    min_y = min(p[1] for p in p_points) + min(q[1] for q in q_points)
    origin_x = min_x - 1
    origin_y = min_y - 1
    origin = (origin_x, origin_y)
    
    p_angles = []
    for i in range(n):
        angle = math.atan2(p_points[i][1] - origin[1], p_points[i][0] - origin[0])
        p_angles.append((angle, i+1))
    p_angles.sort(key=lambda x: x[0])
    p_sorted_indices = [index for angle, index in p_angles]
    
    q_angles = []
    for i in range(n):
        angle = math.atan2(q_points[i][1] - origin[1], q_points[i][0] - origin[0])
        q_angles.append((angle, i+1))
    q_angles.sort(key=lambda x: x[0])
    q_sorted_indices = [index for angle, index in q_angles]
    
    r = [0] * n
    for i in range(n):
        r[p_sorted_indices[i]-1] = q_sorted_indices[i]
        
    is_valid_permutation = True
    segments = []
    for i in range(n):
        segments.append((p_points[i], q_points[r[i]-1]))
        
    for i in range(n):
        for j in range(i + 1, n):
            if segments_intersect(segments[i], segments[j]):
                is_valid_permutation = False
                break
        if not is_valid_permutation:
            break
            
    if is_valid_permutation:
        print(*(r))
    else:
        print("-1")

if __name__ == '__main__':
    solve()