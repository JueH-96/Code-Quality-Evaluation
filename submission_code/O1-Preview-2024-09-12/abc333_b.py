import sys
import math

def get_coordinates(label):
    angles_deg = {
        'A': 90,
        'B': 18,
        'C': -54,
        'D': -126,
        'E': -198
    }
    angle_deg = angles_deg[label]
    angle_rad = math.radians(angle_deg)
    x = math.cos(angle_rad)
    y = math.sin(angle_rad)
    return x, y

S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

S1, S2 = S[0], S[1]
T1, T2 = T[0], T[1]

x_S1, y_S1 = get_coordinates(S1)
x_S2, y_S2 = get_coordinates(S2)
x_T1, y_T1 = get_coordinates(T1)
x_T2, y_T2 = get_coordinates(T2)

d_S = math.hypot(x_S1 - x_S2, y_S1 - y_S2)
d_T = math.hypot(x_T1 - x_T2, y_T1 - y_T2)

epsilon = 1e-6

if abs(d_S - d_T) < epsilon:
    print('Yes')
else:
    print('No')