# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()
Sx, Sy, Tx, Ty = map(int, data)

def min_toll(Sx, Sy, Tx, Ty):
    Sx, Sy, Tx, Ty = Sx + 0.5, Sy + 0.5, Tx + 0.5, Ty + 0.5
    dx, dy = abs(Sx - Tx), abs(Sy - Ty)
    if (Sx + Sy) % 2 == (Tx + Ty) % 2:
        return int(dx + dy)
    else:
        return int(dx + dy - min(dx, dy))

print(min_toll(Sx, Sy, Tx, Ty))