def ccw(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def intersect(ax, ay, bx, by, cx, cy, dx, dy):
    if max(ax, bx) < min(cx, dx) or max(cx, dx) < min(ax, bx):
        return False
    if max(ay, by) < min(cy, dy) or max(cy, dy) < min(ay, by):
        return False
    
    abc = ccw(ax, ay, bx, by, cx, cy)
    abd = ccw(ax, ay, bx, by, dx, dy)
    cda = ccw(cx, cy, dx, dy, ax, ay)
    cdb = ccw(cx, cy, dx, dy, bx, by)
    
    return abc * abd <= 0 and cda * cdb <= 0 and not (abc == 0 and abd == 0)

def check_intersection(i, j, perm, points_p, points_q):
    pi_x, pi_y = points_p[i]
    qi_x, qi_y = points_q[perm[i]-1]
    pj_x, pj_y = points_p[j]
    qj_x, qj_y = points_q[perm[j]-1]
    
    return intersect(pi_x, pi_y, qi_x, qi_y, pj_x, pj_y, qj_x, qj_y)

def solve(n, points_p, points_q, used, perm):
    if len(perm) == n:
        for i in range(n):
            for j in range(i+1, n):
                if check_intersection(i, j, perm, points_p, points_q):
                    return None
        return perm
    
    curr = len(perm)
    for i in range(n):
        if not used[i]:
            perm.append(i+1)
            used[i] = True
            
            valid = True
            for j in range(curr):
                if check_intersection(j, curr, perm, points_p, points_q):
                    valid = False
                    break
                    
            if valid:
                result = solve(n, points_p, points_q, used, perm)
                if result is not None:
                    return result
                    
            perm.pop()
            used[i] = False
            
    return None

n = int(input())
points_p = []
points_q = []

for _ in range(n):
    a, b = map(int, input().split())
    points_p.append((a, b))
    
for _ in range(n):
    c, d = map(int, input().split())
    points_q.append((c, d))

used = [False] * n
result = solve(n, points_p, points_q, used, [])

if result is None:
    print(-1)
else:
    print(*result)