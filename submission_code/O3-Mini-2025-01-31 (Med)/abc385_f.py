# YOUR CODE HERE
from math import atan
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    # one building: no blocking possible so always full visibility
    if n == 1:
        print(-1)
        return
    buildings = []
    pos = 1
    for i in range(n):
        X = int(data[pos]); H = int(data[pos+1])
        pos += 2
        angle = atan(H / X)
        buildings.append((angle, X, H))
        
    # sort by angle (and then by X in case of ties)
    buildings.sort(key=lambda tup: (tup[0], tup[1]))
    
    # Compute the maximum critical value D; for two consecutive buildings a and b,
    # candidate = (H_a * X_b - H_b * X_a)/(X_b - X_a)
    D = -1e18
    for i in range(len(buildings)-1):
        _, X1, H1 = buildings[i]
        _, X2, H2 = buildings[i+1]
        diff = X2 - X1  # positive because all X's are distinct
        candidate = (H1 * X2 - H2 * X1) / diff
        if candidate > D:
            D = candidate
    if D < 0:
        print(-1)
    else:
        print(f"{D:.18f}")

if __name__ == '__main__':
    main()