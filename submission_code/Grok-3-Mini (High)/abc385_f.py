import sys

def cross(p, q, r):
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - q[0]
    dy2 = r[1] - q[1]
    return dx1 * dy2 - dy1 * dx2

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
points = []
for _ in range(N):
    X = int(data[index])
    H = int(data[index + 1])  # height, use as y
    points.append((X, H))
    index += 2

if N < 2:
    print(-1)
    sys.exit()

# Sort points by X
points.sort()

# Compute lower hull
lower = []
for p in points:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
        lower.pop()
    lower.append(p)

# Compute upper hull
upper = []
up_points = list(reversed(points))  # reversed copy
for p in up_points:
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
        upper.pop()
    upper.append(p)

# Get all hull vertices
hull_set = set(lower) | set(upper)  # set of tuples
hull_list = list(hull_set)
# Sort by X
hull_list.sort()

# Compute h for consecutive pairs in hull_list
max_h = float('-inf')
for i in range(len(hull_list) - 1):
    p_left = hull_list[i]
    p_right = hull_list[i + 1]
    X_l, H_l = p_left
    X_r, H_r = p_right
    num = (X_r * H_l - X_l * H_r)
    den = X_r - X_l  # positive since X increasing
    h_val = num / den  # float division
    if h_val > max_h:
        max_h = h_val

# Output the result
if max_h >= 0:
    print(f"{max_h:.15f}")
else:
    print(-1)