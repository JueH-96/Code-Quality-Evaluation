import math

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

N = int(input())
P = []
Q = []

for i in range(N):
    a, b = map(int, input().split())
    P.append((a, b))

for i in range(N):
    c, d = map(int, input().split())
    Q.append((c, d))

# Calculate centroid of all points
all_points = P + Q
cx = sum(p[0] for p in all_points) / (2 * N)
cy = sum(p[1] for p in all_points) / (2 * N)
center = (cx, cy)

# Sort P and Q by angle from centroid
def angle_key(point):
    return math.atan2(point[1] - center[1], point[0] - center[0])

# Create lists with indices
P_with_idx = [(i, P[i]) for i in range(N)]
Q_with_idx = [(i, Q[i]) for i in range(N)]

# Sort by angle
P_with_idx.sort(key=lambda x: angle_key(x[1]))
Q_with_idx.sort(key=lambda x: angle_key(x[1]))

# Create permutation
R = [0] * N
for i in range(N):
    p_idx = P_with_idx[i][0]
    q_idx = Q_with_idx[i][0]
    R[p_idx] = q_idx + 1

# Verify the solution
valid = True
for i in range(N):
    for j in range(i + 1, N):
        if do_intersect(P[i], Q[R[i] - 1], P[j], Q[R[j] - 1]):
            valid = False
            break
    if not valid:
        break

if valid:
    print(' '.join(map(str, R)))
else:
    # Try different approach: sort by x-coordinate
    P_with_idx.sort(key=lambda x: (x[1][0], x[1][1]))
    Q_with_idx.sort(key=lambda x: (x[1][0], x[1][1]))
    
    R = [0] * N
    for i in range(N):
        p_idx = P_with_idx[i][0]
        q_idx = Q_with_idx[i][0]
        R[p_idx] = q_idx + 1
    
    print(' '.join(map(str, R)))